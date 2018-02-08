source cmd.sh
source path.sh
#SETSHELL grid
#SETSHELL sptk

# This recipe uses the arctic database as audio input.
# Please feel free to adapt it to other DB

#WSJCAM0=/idiap/resource/database/WSJCAM0/
#WSJAUDIO=$WSJCAM0/audio/data/primary_microphone/
#WSJLABEL=$WSJCAM0/dbase/data/primary_microphone/
#TMPDIR=/idiap/temp/$USER


#### General instructions for training a DNN-TTS system ####
# 1. Generate acoustic features with emime script
# 2. Generate full / mono labels from festival
# 3. Align these labels with audio: create fake dictionary where phones are mapped to phones,
#    replace text with the phones of mono labels. Use normal kaldi setup on the MFCC features
#    only (i.e. the first 40 scalar of acoustic vector)
# 4. Generate acoustic / label dataset from above data
# 5. Train the DNN :-)

####################################
##### Step 0: data preparation #####
####################################

# Arctic database usual sampling rate; although 32k is
# also available for some speakers.
# upgrade to using 48k data from HTS demo
srate=48000
FRAMESHIFT=0.005
#TMPDIR=/tmp
TMPDIR=/share/cere01/potard
nj=4
# Clean up
rm -rf data/train data/eval data/dev data/train_* data/eval_* data/dev_* data/full

# Speaker ID
spks="slt" # can be any of slt, bdl, jmk

echo "##### Step 0: data preparation #####"
mkdir -p data/{train,dev}
for spk in $spks; do
    # URL of arctic DB
    arch=cmu_us_${spk}_arctic-WAVEGG.tar.bz2
    url=http://festvox.org/cmu_arctic/cmu_arctic/orig/$arch
    laburl=http://festvox.org/cmu_arctic/cmuarctic.data
    audio_dir=rawaudio/cmu_us_${spk}_arctic/48k
    #audio_dir_16k=rawaudio/cmu_us_${spk}_arctic/16k
    label_dir=labels/cmu_us_${spk}_arctic
    # Download data
    if [ ! -e $audio_dir ]; then
	    mkdir -p rawaudio
	    cd rawaudio
	    wget -N -c $url
	    tar xjf $arch
	    cd ..
        mkdir -p $audio_dir 
        for i in rawaudio/cmu_us_${spk}_arctic/orig/*.wav; do
            # sox $i -r 48k $audio_dir/`basename $i` remix 1 upsample 2
            # sox $i $audio_dir/`basename $i` remix 1 rate -v -s -a 48000 dither -s
            sox $i $audio_dir/`basename $i` remix 1 rate -v -s -a 48000 dither -s
            #sox $i $audio_dir_16k/`basename $i` remix 1 rate -v -s -a 16000 dither -s
        done
    fi
    if [ ! -e $label_dir ]; then
	    mkdir -p $label_dir
	    cd $label_dir
	    wget $laburl
	    cd ../..
    fi

    # Create train, dev sets
    dev_pat='arctic_a0??2'
    dev_rgx='arctic_a0..2'
    train_pat='arctic_?????'
    train_rgx='arctic_.....'

    makeid="xargs -n 1 -I {} -n 1 awk -v lst={} -v spk=$spk BEGIN{print(gensub(\".*/([^.]*)[.].*\",spk\"_\\\\1\",\"g\",lst),lst)}"
    makelab="awk -v spk=$spk '{u=\$2;\$1=\"\";\$2=\"\";\$NF=\"\";print(\"<fileid id=\\\"\" spk \"_\" u \"\\\">\",substr(\$0,4,length(\$0)-5),\"</fileid>\")}'"

    find $audio_dir -iname "$train_pat".wav  | grep -v "$dev_rgx" | sort | $makeid >> data/train/wav.scp
    find $audio_dir -iname "$dev_pat".wav    | sort | $makeid >> data/dev/wav.scp

    grep "$train_rgx" $label_dir/cmuarctic.data | grep -v "$dev_rgx" | sort | eval "$makelab" >> data/train/text.xml
    grep "$dev_rgx"   $label_dir/cmuarctic.data | sort | eval "$makelab" >> data/dev/text.xml

    # Generate utt2spk / spk2utt info
    # TODO: ensure train / dev list are disjoint
    for step in train dev; do
	    cat data/$step/wav.scp | awk -v spk=$spk '{print $1, spk}' >> data/$step/utt2spk
	    utt2spk_to_spk2utt.pl < data/$step/utt2spk > data/$step/spk2utt
    done
done

mkdir -p data/full
# This will be used for aligning the labels only
for k in text.xml wav.scp utt2spk; do
    cat data/{train,dev}/$k | sort -u > data/full/$k
done
utt2spk_to_spk2utt.pl < data/full/utt2spk > data/full/spk2utt

# Turn transcription into valid xml files
for step in full train dev; do
    cat data/$step/text.xml | awk 'BEGIN{print "<document>"}{print}END{print "</document>"}' > data/$step/text2
    mv data/$step/text2 data/$step/text.xml
done

############################################
##### Step 1: acoustic data generation #####
############################################

echo "##### Step 1: acoustic data generation #####"
export featdir=$TMPDIR/dnn_feats/arctic2

# Use kaldi to generate MFCC features for alignment
for step in full; do
    steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc-48k.conf data/$step exp/make_mfcc/$step $featdir
    steps/compute_cmvn_stats.sh data/$step exp/make_mfcc/$step $featdir
done

# Use Kaldi + SPTK tools to generate F0 / BNDAP / MCEP
# NB: respective configs are in conf/pitch.conf, conf/bndap.conf, conf/mcep.conf
for step in dev train; do
    rm -f data/$step/feats.scp
    # Generate f0 features
    steps/make_pitch.sh --nj $nj --pitch-config conf/pitch-48k.conf  data/$step    exp/make_pitch/$step   $featdir || exit 1;
    cp data/$step/pitch_feats.scp data/$step/feats.scp
    # Compute CMVN on pitch features, to estimate min_f0 (set as mean_f0 - 2*std_F0)
    steps/compute_cmvn_stats.sh data/$step    exp/compute_cmvn_pitch/$step   $featdir || exit 1;
    # For bndap / mcep extraction to be successful, the frame-length must be adjusted
    # in relation to the minimum pitch frequency.
    # We therefore do something speaker specific using the mean / std deviation from
    # the pitch for each speaker.
    for spk in $spks; do
	    min_f0=`copy-feats scp:"awk -v spk=$spk '(\\$1 == spk){print}' data/$step/cmvn.scp |" ark,t:- \
	    | awk '(NR == 2){n = \$NF; m = \$2 / n}(NR == 3){std = sqrt(\$2/n - m * m)}END{print m - 2*std}'`
	    echo $min_f0
	    # Rule of thumb recipe; probably try with other window sizes?
	    bndapflen=`awk -v f0=$min_f0 'BEGIN{printf "%d", 4.6 * 1000.0 / f0 + 0.5}'`
	    mcepflen=`awk -v f0=$min_f0 'BEGIN{printf "%d", 2.3 * 1000.0 / f0 + 0.5}'`
	    f0flen=`awk -v f0=$min_f0 'BEGIN{printf "%d", 2.3 * 1000.0 / f0 + 0.5}'`
	    echo "using wsizes: $bndapflen $mcepflen"
	    subset_data_dir.sh --spk $spk data/$step 100000 data/${step}_$spk
	    #cp data/$step/pitch_feats.scp data/${step}_$spk/
	    # Regenerate pitch with more appropriate window
	    steps/make_pitch.sh --nj $nj --pitch-config conf/pitch-48k.conf --frame_length $f0flen    data/${step}_$spk exp/make_pitch/${step}_$spk  $featdir || exit 1;
	    # Generate Band Aperiodicity feature
	    steps/make_bndap.sh --nj $nj --bndap-config conf/bndap-48k.conf --frame_length $bndapflen data/${step}_$spk exp/make_bndap/${step}_$spk  $featdir || exit 1;
        # Generate LSF features
        #steps/make_lsf.sh   --lsf-config conf/lsf-48k.conf  data/${step}_$spk exp/make_lsf/${step}_$spk   $featdir   || exit 1;
	    # Generate Mel Cepstral features
	    steps/make_mcep.sh  --nj $nj --mcep-config conf/mcep-48k.conf --frame_length $mcepflen  data/${step}_$spk exp/make_mcep/${step}_$spk   $featdir   || exit 1;
    done
    # Merge features
    cat data/${step}_*/bndap_feats.scp > data/$step/bndap_feats.scp
    cat data/${step}_*/mcep_feats.scp > data/$step/mcep_feats.scp
    #cat data/${step}_*/lsf_feats.scp > data/$step/lsf_feats.scp
    # Have to set the length tolerance to 1, as mcep files are a bit longer than the others for some reason
    paste-feats --length-tolerance=2 scp:data/$step/pitch_feats.scp scp:data/$step/mcep_feats.scp scp:data/$step/bndap_feats.scp ark,scp:$featdir/${step}_cmp_feats.ark,data/$step/feats.scp
    # Compute CMVN on whole feature set
    steps/compute_cmvn_stats.sh data/$step exp/compute_cmvn/$step   data/$step || exit 1;
done

#cat data/{train,dev}/feats.scp | awk '{ if (w[$1]){} else {w[$1] = 1; print}}' > data/full/feats.scp

############################################
#####      Step 2: label creation      #####
############################################

echo "##### Step 2: label creation #####"
# We are using the idlak front-end for processing the text
tpdb=$KALDI_ROOT/idlak-data/en/ga/
dict=data/local/dict
for step in train dev full; do
    # Normalise text and generate phoneme information
    idlaktxp --pretty --tpdb=$tpdb data/$step/text.xml data/$step/text_norm.xml || exit 1;
    # Generate full labels
    #idlakcex --pretty --cex-arch=default --tpdb=$tpdb data/$step/text_norm.xml data/$step/text_full.xml
done
# Generate language models for alignment
mkdir -p $dict
# Create dictionary and text files
python $KALDI_ROOT/idlak-voice-build/utils/idlak_make_lang.py --mode 0 data/full/text_norm.xml data/full $dict
# Fix data directory, in case some recordings are missing
utils/fix_data_dir.sh data/full

#######################################
## 3a: create kaldi forced alignment ##
#######################################

echo "##### Step 3: forced alignment #####"
lang=data/lang
rm -rf $dict/lexiconp.txt $lang
utils/prepare_lang.sh --num-nonsil-states 5 --share-silence-phones true $dict "<OOV>" data/local/lang_tmp $lang
#utils/validate_lang.pl $lang

# Now running the normal kaldi recipe for forced alignment
expa=exp-align
train=data/full
#test=data/eval_mfcc

# By default kaldi splits files per speaker when aligning;
# "manually" split per utterance 
split_data.sh --per-utt $train $nj
[ -d $train/split$nj ] || mv $train/split${nj}utt $train/split$nj
steps/train_mono.sh  --nj $nj --cmd "$train_cmd" \
  $train $lang $expa/mono || exit 1;
steps/align_si.sh  --nj $nj --cmd "$train_cmd" \
$train $lang $expa/mono $expa/mono_ali
steps/train_deltas.sh --cmd "$train_cmd" \
     5000 50000 $train $lang $expa/mono_ali $expa/tri1 || exit 1;

steps/align_si.sh  --nj $nj --cmd "$train_cmd" \
$train data/lang $expa/tri1 $expa/tri1_ali || exit 1;
steps/train_deltas.sh --cmd "$train_cmd" \
     5000 50000 $train $lang $expa/tri1_ali $expa/tri2 || exit 1;

# Create triphone alignments
steps/align_si.sh  --nj $nj --cmd "$train_cmd" \
    $train $lang $expa/tri2 $expa/tri2_ali_full || exit 1;

steps/train_deltas.sh --cmd "$train_cmd" \
    --context-opts "--context-width=5 --central-position=2" \
    5000 50000 $train $lang $expa/tri2_ali_full $expa/quin || exit 1;

# Create final alignments
#split_data.sh --per-utt $train 9
steps/align_si.sh  --nj $nj --cmd "$train_cmd" \
    $train $lang $expa/quin $expa/quin_ali_full || exit 1;



################################
## 3b. Align with full labels ##
################################

# Convert to phone-state alignement
for step in full; do
    ali=$expa/quin_ali_$step
    # Extract phone alignment
    ali-to-phones --per-frame $ali/final.mdl ark:"gunzip -c $ali/ali.{1..$nj}.gz|" ark,t:- \
	| utils/int2sym.pl -f 2- $lang/phones.txt > $ali/phones.txt
    # Extract state alignment
    ali-to-hmmstate $ali/final.mdl ark:"gunzip -c $ali/ali.{1..$nj}.gz|" ark,t:$ali/states.tra
    # Extract word alignment
    linear-to-nbest ark:"gunzip -c $ali/ali.{1..$nj}.gz|" \
	ark:"utils/sym2int.pl --map-oov 1669 -f 2- $lang/words.txt < data/$step/text |" '' '' ark:- \
	| lattice-align-words $lang/phones/word_boundary.int $ali/final.mdl ark:- ark:- \
	| nbest-to-ctm --frame-shift=$FRAMESHIFT --precision=3 ark:- - \
	| utils/int2sym.pl -f 5 $lang/words.txt > $ali/wrdalign.dat
    
    # Regenerate text output from alignment
    python $KALDI_ROOT/idlak-voice-build/utils/idlak_make_lang.py --mode 1 "2:0.03,3:0.2" "4" $ali/phones.txt $ali/wrdalign.dat data/$step/text_align.xml $ali/states.tra

    # Generate corresponding quinphone full labels
    idlaktxp --pretty --tpdb=$tpdb data/$step/text_align.xml data/$step/text_anorm.xml || exit 1;
    idlakcex --pretty --cex-arch=default --tpdb=$tpdb data/$step/text_anorm.xml data/$step/text_afull.xml || exit 1;
    python $KALDI_ROOT/idlak-voice-build/utils/idlak_make_lang.py --mode 2 data/$step/text_afull.xml data/$step/cex.ark > data/$step/cex_output_dump
    
    # Merge alignment with output from idlak cex front-end => gives you a nice vector
    # NB: for triphone alignment:
    # make-fullctx-ali-dnn  --phone-context=3 --mid-context=1 --max-sil-phone=15 $ali/final.mdl ark:"gunzip -c $ali/ali.*.gz|" ark,t:data/$step/cex.ark ark,t:data/$step/ali
    make-fullctx-ali-dnn --max-sil-phone=15 $ali/final.mdl ark:"gunzip -c $ali/ali.{1..$nj}.gz|" ark,t:data/$step/cex.ark ark,t:data/$step/ali


    # UGLY convert alignment to features
    mkdir -p lbldata/$step
    cat data/$step/ali \
	| awk '{print $1, "["; $1=""; na = split($0, a, ";"); for (i = 1; i < na; i++) print a[i]; print "]"}' \
	| copy-feats ark:- ark,scp:$featdir/in_feats_$step.ark,lbldata/$step/feats.scp
done

# Split labels in train / dev
for step in train dev; do
    dir=lbldata/$step
    mkdir -p $dir
    #cp data/$step/{utt2spk,spk2utt} $dir
    utils/filter_scp.pl data/$step/utt2spk lbldata/full/feats.scp > $dir/feats.scp
    cat data/$step/utt2spk | awk -v lst=$dir/feats.scp 'BEGIN{ while (getline < lst) n[$1] = 1}{if (n[$1]) print}' > $dir/utt2spk
    utils/utt2spk_to_spk2utt.pl < $dir/utt2spk > $dir/spk2utt
    #steps/compute_cmvn_stats.sh $dir $dir $dir
done

# Optional: remove silences
for step in train dev; do
    local/remove_silence_frames.sh data/$step lbldata/$step
done
# Cancel silence removal
#for step in train dev; do
#    for d in data lbldata; do
#        cp $d/$step/spk2utt-sil $d/$step/spk2utt
#        cp $d/$step/utt2spk-sil $d/$step/utt2spk
#        cp $d/$step/feats-sil.scp $d/$step/feats.scp 
#    done
#done

for step in train dev; do
    dir=lbldata/$step
    steps/compute_cmvn_stats.sh $dir $dir $dir
done

for f in feats.scp utt2spk; do
    cat lbldata/{train,dev}/$f | sort > lbldata/full/$f
done
utils/utt2spk_to_spk2utt.pl < lbldata/full/utt2spk > lbldata/full/spk2utt


# HACKY
# Generate features for duration modelling
# we remove relative position within phone and state
copy-feats scp:lbldata/full/feats.scp ark,t:- \
    | awk -v nstate=5 'BEGIN{oldkey = 0; oldstate = -1; for (s = 0; s < nstate; s++) asd[s] = 0}
function print_phone(vkey, vasd, vpd) {
      for (s = 0; s < nstate; s++) {
         print vkey, s, vasd[s], vpd;
         vasd[s] = 0;
      }
}
(NF == 2){print}
(NF > 2){
   n = NF; 
   if ($NF == "]") n = NF - 1;
   state = $(n-4); sd = $(n-3); pd = $(n-1);
   for (i = n-4; i <= NF; i++) $i = "";
   len = length($0);
   if (n != NF) len = len -1;
   key = substr($0, 1, len - 5);
   if ((key != oldkey) && (oldkey != 0)) {
      print_phone(oldkey, asd, opd);
      oldstate = -1;
   }
   if (state != oldstate) {
      asd[state] += sd;
   }
   opd = pd;
   oldkey = key;
   oldstate = state;
   if (NF != n) {
      print_phone(key, asd, opd);
      oldstate = -1;
      oldkey = 0;
      print "]";
   }
}' > $featdir/tmp_durfeats_full.ark

duration_feats="ark:$featdir/tmp_durfeats_full.ark"
nfeats=$(feat-to-dim "$duration_feats" -)
# Input 
select-feats 0-$(( $nfeats - 3 )) "$duration_feats" ark,scp:$featdir/in_durfeats_full.ark,$featdir/in_durfeats_full.scp
# Output: duration of phone and state are assumed to be the 2 last features
select-feats $(( $nfeats - 2 ))-$(( $nfeats - 1 )) "$duration_feats" ark,scp:$featdir/out_durfeats_full.ark,$featdir/out_durfeats_full.scp

# Split labels for duration
for step in train dev; do
    dir=lbldurdata/$step
    mkdir -p $dir
    #cp data/$step/{utt2spk,spk2utt} $dir
    utils/filter_scp.pl data/$step/utt2spk $featdir/in_durfeats_full.scp > $dir/feats.scp
    cat data/$step/utt2spk | awk -v lst=$dir/feats.scp 'BEGIN{ while (getline < lst) n[$1] = 1}{if (n[$1]) print}' > $dir/utt2spk
    utils/utt2spk_to_spk2utt.pl < $dir/utt2spk > $dir/spk2utt
    steps/compute_cmvn_stats.sh $dir $dir $dir

    dir=durdata/$step
    mkdir -p $dir
    #cp data/$step/{utt2spk,spk2utt} $dir
    utils/filter_scp.pl data/$step/utt2spk $featdir/out_durfeats_full.scp > $dir/feats.scp
    cat data/$step/utt2spk | awk -v lst=$dir/feats.scp 'BEGIN{ while (getline < lst) n[$1] = 1}{if (n[$1]) print}' > $dir/utt2spk
    utils/utt2spk_to_spk2utt.pl < $dir/utt2spk > $dir/spk2utt
    steps/compute_cmvn_stats.sh $dir $dir $dir
done

acdir=data
lbldir=lbldata

#ensure consistency in lists
#for dir in $lbldir $acdir; do
for class in train dev; do
    cp $lbldir/$class/feats.scp $lbldir/$class/feats_full.scp
    cp $acdir/$class/feats.scp $acdir/$class/feats_full.scp
    cat $acdir/$class/feats_full.scp | awk -v lst=$lbldir/$class/feats_full.scp 'BEGIN{ while (getline < lst) n[$1] = 1}{if (n[$1]) print}' > $acdir/$class/feats.scp
    cat $lbldir/$class/feats_full.scp | awk -v lst=$acdir/$class/feats_full.scp 'BEGIN{ while (getline < lst) n[$1] = 1}{if (n[$1]) print}' > $lbldir/$class/feats.scp
done

##############################
## 4. Train DNN
##############################

echo "##### Step 4: training DNNs #####"

exp=exp_dnn
mkdir -p $exp

# Very basic one for testing
#mkdir -p $exp
#dir=$exp/tts_dnn_train_3e
#$cuda_cmd $dir/_train_nnet.log steps/train_nnet_basic.sh --config conf/3-layer-nn.conf --learn_rate 0.2 --momentum 0.1 --halving-factor 0.5 --min_iters 15 --randomize true --bunch_size 50 --mlpOption " " --hid-dim 300 $lbldir/train $lbldir/dev $acdir/train $acdir/dev $dir

echo " ### Step 4a: duration model DNN ###"
# A. Small one for duration modelling
durdir=durdata
lbldurdir=lbldurdata
expdurdir=$exp/tts_dnn_dur_3_delta_quin5
rm -rf $expdurdir
$cuda_cmd $expdurdir/_train_nnet.log steps/train_nnet_basic.sh --delta_order 2 --config conf/3-layer-nn-splice5.conf --learn_rate 0.02 --momentum 0.1 --halving-factor 0.5 --min_iters 15 --max_iters 50 --randomize true --cache_size 50000 --bunch_size 200 --mlpOption " " --hid-dim 100 $lbldurdir/train $lbldurdir/dev $durdir/train $durdir/dev $expdurdir || exit 1;

# B. Larger DNN for acoustic features
echo " ### Step 4b: acoustic model DNN ###"

dnndir=$exp/tts_dnn_train_3_deltasc2_quin5
rm -rf $dnndir
$cuda_cmd $dnndir/_train_nnet.log steps/train_nnet_basic.sh --delta_order 2 --config conf/3-layer-nn-splice5.conf --learn_rate 0.04 --momentum 0.1 --halving-factor 0.5 --min_iters 15 --randomize true --cache_size 50000 --bunch_size 200 --mlpOption " " --hid-dim 700 $lbldir/train $lbldir/dev $acdir/train $acdir/dev $dnndir || exit 1;

##############################
## 5. Synthesis
##############################

if [ "$srate" == "16000" ]; then
    order=39
    alpha=0.42
    fftlen=1024
    bndap_order=21
elif [ "$srate" == "48000" ]; then
    order=60
    alpha=0.55
    fftlen=4096
    bndap_order=25
fi

echo "##### Step 5: synthesis #####"
# Original samples:
echo "Synthesizing vocoded training samples"
mkdir -p exp_dnn/orig2/cmp exp_dnn/orig2/wav
copy-feats scp:data/dev/feats.scp ark,t:- | awk -v dir=exp_dnn/orig2/cmp/ '($2 == "["){if (out) close(out); out=dir $1 ".cmp";}($2 != "["){if ($NF == "]") $NF=""; print $0 > out}'
for cmp in exp_dnn/orig2/cmp/*.cmp; do
    utils/mlsa_synthesis_63_mlpg.sh --voice_thresh 0.5 --alpha $alpha --fftlen $fftlen --srate $srate --bndap_order $bndap_order --mcep_order $order $cmp exp_dnn/orig2/wav/`basename $cmp .cmp`.wav
done

# Variant with mlpg: requires mean / variance from coefficients
copy-feats scp:data/train/feats.scp ark:- \
    | add-deltas --delta-order=2 ark:- ark:- \
    | compute-cmvn-stats --binary=false ark:- - \
    | awk '
(NR==2){count=$NF; for (i=1; i < NF; i++) mean[i] = $i / count}
(NR==3){if ($NF == "]") NF -= 1; for (i=1; i < NF; i++) var[i] = $i / count - mean[i] * mean[i]; nv = NF-1}
END{for (i = 1; i <= nv; i++) print mean[i], var[i]}' \
    > data/train/var_cmp.txt

echo "  ###  5b: Alice samples synthesis ###"
# Alice test set
mkdir -p data/eval
cp $KALDI_ROOT/idlak-data/en/testdata/alice.xml data/eval/text.xml

# Generate CEX features for test set.
for step in eval; do
    idlaktxp --pretty --tpdb=$tpdb data/$step/text.xml - \
     | idlakcex --pretty --cex-arch=default --tpdb=$tpdb - data/$step/text_full.xml
    python $KALDI_ROOT/idlak-voice-build/utils/idlak_make_lang.py --mode 2 -r "alice" \
        data/$step/text_full.xml data/full/cex.ark.freq data/$step/cex.ark > data/$step/cex_output_dump
    # Generate input feature for duration modelling
    cat data/$step/cex.ark \
        | awk '{print $1, "["; $1=""; na = split($0, a, ";"); for (i = 1; i < na; i++) for (state = 0; state < 5; state++) print a[i], state; print "]"}' \
        | copy-feats ark:- ark,scp:$featdir/in_durfeats_$step.ark,$featdir/in_durfeats_$step.scp
done

# Duration based test set
for step in eval; do
    dir=lbldurdata/$step
    mkdir -p $dir
    cp $featdir/in_durfeats_$step.scp $dir/feats.scp
    cut -d ' ' -f 1 $dir/feats.scp | awk -v spk=$spk '{print $1, spk}' > $dir/utt2spk
    utils/utt2spk_to_spk2utt.pl $dir/utt2spk > $dir/spk2utt
    steps/compute_cmvn_stats.sh $dir $dir $dir
done

# Generate label with DNN-generated duration
echo "Synthesizing MLPG eval samples"
#  1. forward pass through duration DNN
utils/make_forward_fmllr.sh $expdurdir $lbldurdir/eval $expdurdir/tst_forward/ ""
#  2. make the duration consistent, generate labels with duration information added
(echo '#!MLF!#'; for cmp in $expdurdir/tst_forward/cmp/*.cmp; do
    cat $cmp | awk -v nstate=5 -v id=`basename $cmp .cmp` 'BEGIN{print "\"" id ".lab\""; tstart = 0 }
{
  pd += $2;
  sd[NR % nstate] = $1}
(NR % nstate == 0){
   mpd = pd / nstate;
   smpd = 0;
   for (i = 1; i <= nstate; i++) smpd += sd[i % nstate];
   rmpd = int((smpd + mpd) / 2 + 0.5);
   # Normal phones => state 5 has a duration of 0
   if (int(sd[0] + 0.5) == 0) {
      for (i = 1; i <= 3; i++) {
         sd[i % nstate] = int(sd[i % nstate] / smpd * rmpd + 0.5);
      }
      if (sd[3] <= 0) sd[3] = 1;
      for (i = 4; i <= nstate; i++) sd[i % nstate] = 0;
   }
   # Silence phone
   else {
      for (i = 1; i <= nstate; i++) {
          sd[i % nstate] = int(sd[i % nstate] / smpd * rmpd + 0.5);
      }
      if (sd[0] <= 0) sd[0] = 1;
   }
   if (sd[1] <= 0) sd[1] = 1;
   smpd = 0;
   for (i = 1; i <= nstate; i++) smpd += sd[i % nstate];
   for (i = 1; i <= nstate; i++) {
        if (sd[i % nstate] > 0) {
           tend = tstart + sd[i % nstate] * 50000;
           print tstart, tend, int(NR / 5), i-1;
           tstart = tend;
        }
   }
   pd = 0;
}' 
done) > data/eval/synth_lab.mlf
# 3. Turn them into DNN input labels (i.e. one sample per frame)
for step in eval; do
    python utils/make_fullctx_mlf_dnn.py data/$step/synth_lab.mlf data/$step/cex.ark data/$step/feat.ark
    copy-feats ark:data/$step/feat.ark ark,scp:$featdir/in_feats_$step.ark,$featdir/in_feats_$step.scp
done
for step in eval; do
    dir=lbldata/$step
    mkdir -p $dir
    cp $featdir/in_feats_$step.scp $dir/feats.scp
    cut -d ' ' -f 1 $dir/feats.scp | awk -v spk=$spk '{print $1, spk}' > $dir/utt2spk
    utils/utt2spk_to_spk2utt.pl $dir/utt2spk > $dir/spk2utt
    steps/compute_cmvn_stats.sh $dir $dir $dir
done
# 4. Forward pass through big DNN
utils/make_forward_fmllr.sh $dnndir $lbldir/eval $dnndir/tst_forward/ ""

# 5. Vocoding
# NB: these are the settings for 16k
mkdir -p $dnndir/tst_forward/wav_mlpg/; for cmp in $dnndir/tst_forward/cmp/*.cmp; do
    utils/mlsa_synthesis_63_mlpg.sh --voice_thresh 0.5 --alpha $alpha --fftlen $fftlen --srate $srate --bndap_order $bndap_order --mcep_order $order --delta_order 2 $cmp $dnndir/tst_forward/wav_mlpg/`basename $cmp .cmp`.wav data/train/var_cmp.txt
done

echo "
*********************
** Congratulations **
*********************
TTS-DNN trained and sample synthesis done.

Samples can be found in $dnndir/tst_forward/wav_mlpg/*.wav.

More synthesis can be performed using the utils/synthesis_test.sh utility,
e.g.: echo 'Test 1 2 3' | utils/synthesis_test-48k.sh
"
echo "#### Step 6: packaging DNN voice ####"

utils/make_dnn_voice.sh --spk $spk --srate $srate --mcep_order $order --bndap_order $bndap_order --alpha $alpha --fftlen $fftlen 

echo "Voice packaged successfully. Portable models have been stored in ${spk}_mdl."
echo "Synthesis can be performed using: 
         echo \"This is a demo of D N N synthesis\" | utils/synthesis_voice.sh ${spk}_mdl <outdir>"


