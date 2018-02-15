source cmd.sh
source path.sh

####################################
##### Step 0: data preparation #####
####################################

srate=48000
FRAMESHIFT=0.005
TMPDIR=/tmp
stage=-1
endstage=7
nj=4 # max 9
# Speaker ID
spks=LJ 
network_type=dnn # dnn or lstm

# Download and extract LJSpeech, then set the following to point to
# audio and labels 
audio_input_dir=/share/resources/LJSpeech-1.0/wavs/
label_input_file=/share/resources/LJSpeech-1.0/metadata.csv

. parse_options.sh || exit 1;

if [ ! -e $label_input_file ]; then
    echo "Please download the LJSpeech corpus from https://keithito.com/LJ-Speech-Dataset/,"
    echo "extract it, then set the audio_input_dir and label_input_file variables to point"
    echo "to the locations of the audio directory and labels file, respectively."
fi

function incr_stage(){
   stage=$(( $stage + 1 ))
   if [ $stage -gt $endstage ]; then
       exit 0
   fi
}


# Clean up
if [ $stage -le -1 ]; then
    rm -rf data/train data/eval data/dev data/train_* data/eval_* data/dev_* data/full
    stage=0
fi
spk=$spks


## Stage 0: Extract audio
if [ $stage -le 0 ]; then
    echo -e "\n\n##### Step 0: data preparation #####\n"
    mkdir -p data/{train,dev}
    audio_dir=rawaudio/ljspeech/$srate
    label_dir=labels/ljspeech
    if [ ! -e $audio_dir ]; then
        echo "Converting audio files..."
        mkdir -p $audio_dir
        for i in `find $audio_input_dir -iname '*.wav' | sort`; do
            outi=`echo $i| sed "s|$audio_input_dir||; s|/|_|g"`
            echo "$i => $audio_dir/$outi"
            sox $i $audio_dir/$outi remix 1 rate -v -s -a $srate dither -s
        done
    fi

    

    # Create train, dev sets
    dev_pat='LJ???-0??2'
    dev_rgx='LJ...-0..2'
    train_pat='LJ???-????'
    train_rgx='LJ...-....'

    makeid="xargs -n 1 -I {} -n 1 awk -v lst={} BEGIN{print(gensub(\".*/([^.]*)[.].*\",\"\\\\1\",\"g\",lst),lst)}"
    makelab="awk '{u=\$1;\$1=\"\";print(\"<fileid id=\\\"\" u \"\\\">\",\$0,\"</fileid>\")}'"

    echo "Generating audio lists..."
    find $audio_dir -iname "$train_pat".wav  | grep -v "$dev_rgx" | sort | $makeid > data/train/wav.scp
    find $audio_dir -iname "$dev_pat".wav    | sort | $makeid > data/dev/wav.scp

    echo "Generating labels..."
    cat $label_input_file | awk -v FS="|" '{print $1,$3}' | grep "$train_rgx" | grep -v "$dev_rgx" | sort | eval "$makelab" > data/train/text.xml
    cat $label_input_file | awk -v FS="|" '{print $1,$3}' | grep "$dev_rgx"   | sort | eval "$makelab" > data/dev/text.xml

    # Generate utt2spk / spk2utt info
    for step in train dev; do
	    cat data/$step/wav.scp | awk -v spk=$spk '{print $1, spk}' > data/$step/utt2spk
	    utt2spk_to_spk2utt.pl < data/$step/utt2spk > data/$step/spk2utt
    done
   

    mkdir -p data/full
    for k in text.xml wav.scp utt2spk; do
        cat data/{train,dev}/$k | sort -u > data/full/$k
    done
    utt2spk_to_spk2utt.pl < data/full/utt2spk > data/full/spk2utt
    rm -f data/full/text

    # Turn transcription into valid xml files
    for step in full train dev; do
        cat data/$step/text.xml | awk 'BEGIN{print "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n<document>"}{print}END{print "</document>"}' > data/$step/text2
        mv data/$step/text2 data/$step/text.xml
    done

    incr_stage
fi

export featdir=$TMPDIR/idlak_dnn_feats/ljf


############################################
##### Step 1: acoustic data generation #####
############################################

if [ $stage -le 1 ]; then
    echo -e "\n\n##### Step 1: acoustic data generation #####\n"

    # Use kaldi to generate MFCC features for alignment
    for step in full; do
        steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc-48k.conf data/$step exp/make_mfcc/$step $featdir
        steps/compute_cmvn_stats.sh data/$step exp/make_mfcc/$step $featdir
    done

    # Use Kaldi + SPTK tools to generate F0 / BNDAP / MCEP
    # NB: respective configs are in conf/pitch.conf, conf/bndap.conf, conf/mcep.conf
    for step in train dev; do
        rm -f data/$step/feats.scp
        # Generate f0 features
        steps/make_pitch.sh --nj $nj --pitch-config conf/pitch-48k.conf  data/$step    exp/make_pitch/$step   $featdir;
        cp data/$step/pitch_feats.scp data/$step/feats.scp
        # Compute CMVN on pitch features, to estimate min_f0 (set as mean_f0 - 2*std_F0)
        steps/compute_cmvn_stats.sh data/$step    exp/compute_cmvn_pitch/$step   $featdir;
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
	        steps/make_pitch.sh --nj $nj --pitch-config conf/pitch-48k.conf --frame_length $f0flen    data/${step}_$spk exp/make_pitch/${step}_$spk  $featdir;
	        # Generate Band Aperiodicity feature
	        steps/make_bndap.sh --nj $nj --bndap-config conf/bndap-48k.conf --frame_length $bndapflen data/${step}_$spk exp/make_bndap/${step}_$spk  $featdir
	        # Generate Mel Cepstral features
	        steps/make_mcep.sh --nj $nj --mcep-config  conf/mcep-48k.conf --frame_length $mcepflen  data/${step}_$spk exp/make_mcep/${step}_$spk   $featdir	
        done
        # Merge features
        cat data/${step}_*/bndap_feats.scp > data/$step/bndap_feats.scp
        cat data/${step}_*/mcep_feats.scp > data/$step/mcep_feats.scp
        # Have to set the length tolerance to 1, as mcep files are a bit longer than the others for some reason
        paste-feats --length-tolerance=1 scp:data/$step/mcep_feats.scp scp:data/$step/bndap_feats.scp ark,scp:$featdir/${step}_cmp_feats.ark,data/$step/feats.scp
        # Copy pitch feature in separate folder
        mkdir -p f0data/${step}
        cp data/$step/pitch_feats.scp f0data/${step}/feats.scp
        for k in utt2spk spk2utt; do cp data/$step/$k f0data/${step}/$k; done
        #steps/compute_cmvn_stats.sh data/$step exp/compute_cmvn/$step   data/$step
    done

    incr_stage
fi

#cat data/{train,dev}/feats.scp | awk '{ if (w[$1]){} else {w[$1] = 1; print}}' > data/full/feats.scp

tpdb=$KALDI_ROOT/idlak-data/en/ga/
dict=data/local/dict

############################################
#####      Step 2: label creation      #####
############################################

if [ $stage -le 2 ]; then
    echo -e "\n\n##### Step 2: label creation #####\n"
    # We are using the idlak front-end for processing the text

    for step in train dev full; do
        # Normalise text and generate phoneme information
        idlaktxp --pretty --tpdb=$tpdb data/$step/text.xml data/$step/text_norm.xml
        # Generate full labels
        #idlakcex --pretty --cex-arch=default --tpdb=$tpdb data/$step/text_norm.xml data/$step/text_full.xml
    done
    # Generate language models for alignment
    mkdir -p $dict
    # Create dictionary and text files
    python $KALDI_ROOT/idlak-voice-build/utils/idlak_make_lang.py --mode 0 data/full/text_norm.xml data/full $dict
    # Fix data directory, in case some recordings are missing
    utils/fix_data_dir.sh data/full

    incr_stage
fi

lang=data/lang
expa=exp-align

#######################################
## 3a: create kaldi forced alignment ##
#######################################

if [ $stage -le 3 ]; then
    echo -e "\n\n##### Step 3: forced alignment #####\n"

    rm -rf $dict/lexiconp.txt $lang
    utils/prepare_lang.sh --num-nonsil-states 5 --share-silence-phones true $dict "<OOV>" data/local/lang_tmp $lang
    #utils/validate_lang.pl $lang

    # Now running the normal kaldi recipe for forced alignment
    train=data/full
    #test=data/eval_mfcc

    rm -rf $train/split$nj
    split_data.sh --per-utt $train $nj
    [ -d $train/split$nj ] || mv $train/split${nj}utt $train/split$nj

    steps/train_mono.sh  --nj $nj --cmd "$train_cmd" \
        $train $lang $expa/mono || exit 1;
    steps/align_si.sh  --nj $nj --cmd "$train_cmd" \
        $train $lang $expa/mono $expa/mono_ali || exit 1;
    steps/train_deltas.sh --cmd "$train_cmd" \
        5000 50000 $train $lang $expa/mono_ali $expa/tri1 || exit 1;

    steps/align_si.sh  --nj $nj --cmd "$train_cmd" \
        $train data/lang $expa/tri1 $expa/tri1_ali || exit 1;
    steps/train_deltas.sh --cmd "$train_cmd" \
        5000 50000 $train $lang $expa/tri1_ali $expa/tri2 || exit 1;

    # Create alignments
    steps/align_si.sh  --nj $nj --cmd "$train_cmd" \
        $train $lang $expa/tri2 $expa/tri2_ali_full || exit 1;

    steps/train_deltas.sh --cmd "$train_cmd" \
        --context-opts "--context-width=5 --central-position=2" \
        5000 50000 $train $lang $expa/tri2_ali_full $expa/quin || exit 1;

    # Create alignments
    steps/align_si.sh  --nj $nj --cmd "$train_cmd" \
        $train $lang $expa/quin $expa/quin_ali_full || exit 1;

    incr_stage
fi



################################
## 3b. Align with full labels ##
################################

if [ $stage -le 4 ]; then
    echo -e "\n\n##### Step 4: Align with full labels #####\n"

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
        idlaktxp --pretty --tpdb=$tpdb data/$step/text_align.xml data/$step/text_anorm.xml
        idlakcex --pretty --cex-arch=default --tpdb=$tpdb data/$step/text_anorm.xml data/$step/text_afull.xml
        python $KALDI_ROOT/idlak-voice-build/utils/idlak_make_lang.py --mode 2 data/$step/text_afull.xml data/$step/cex.ark > data/$step/cex_output_dump
        
        # Merge alignment with output from idlak cex front-end => gives you a nice vector
        # NB: for triphone alignment:
        # make-fullctx-ali-dnn  --phone-context=3 --mid-context=1 --max-sil-phone=15 $ali/final.mdl ark:"gunzip -c $ali/ali.*.gz|" ark,t:data/$step/cex.ark ark,t:data/$step/ali
        make-fullctx-ali-dnn --max-sil-phone=15 $ali/final.mdl ark:"gunzip -c $ali/ali.{1..$nj}.gz|" ark,t:data/$step/cex.ark ark,t:data/$step/ali


        # UGLY convert alignment to features
        cat data/$step/ali \
	        | awk '{print $1, "["; $1=""; na = split($0, a, ";"); for (i = 1; i < na; i++) print a[i]; print "]"}' \
	        | copy-feats ark:- ark,scp:$featdir/in_feats_$step.ark,$featdir/in_feats_$step.scp
    done

    # HACKY
    # Generate features for duration modelling
    # we remove relative position within phone and state
    copy-feats ark:$featdir/in_feats_full.ark ark,t:- \
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

    # Split in train / dev
    for step in train dev; do
        dir=lbldata/$step
        mkdir -p $dir
        #cp data/$step/{utt2spk,spk2utt} $dir
        utils/filter_scp.pl data/$step/utt2spk $featdir/in_feats_full.scp > $dir/feats.scp
        cat data/$step/utt2spk | awk -v lst=$dir/feats.scp 'BEGIN{ while (getline < lst) n[$1] = 1}{if (n[$1]) print}' > $dir/utt2spk
        utils/utt2spk_to_spk2utt.pl < $dir/utt2spk > $dir/spk2utt
        steps/compute_cmvn_stats.sh $dir $dir $dir
    done

    # Same for duration
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

    # Same for input of DNN3: pitch + frame-level labels
    # Generate DNN 3 input data: pitch + frames labels
    for step in train dev; do
        dir=lblf0data/$step
        indir=lbldata/$step
        mkdir -p $dir
        paste-feats --length-tolerance=1 scp:data/$step/pitch_feats.scp scp:$indir/feats.scp ark,scp:$featdir/${step}_f0lbl_feats.ark,$dir/feats.scp
        cp $indir/{utt2spk,spk2utt} $dir
        #utils/filter_scp.pl data/$step/utt2spk $featdir/in_durfeats_full.scp > $dir/feats.scp
    done

    incr_stage
fi

acdir=data
lblpitchdir=lblf0data
pitchdir=f0data
lbldir=lbldata
durdir=durdata
lbldurdir=lbldurdata
exp=exp_dnn
mkdir -p $exp
dnndurdir=$exp/tts_${network_type}_dur_3_delta_quin5
dnnf0dir=$exp/tts_${network_type}_f0_3_delta_quin5
dnndir=$exp/tts_${network_type}_train_3_delta_quin5
dnnffdir=$exp/tts_${network_type}_fake_3_delta_quin5

if [ $stage -le 5 ]; then
    #ensure consistency in lists
    #for dir in $lbldir $acdir; do
    for class in train dev; do
        lst=""
        for dir in $acdir $lbldir $pitchdir $lblpitchdir $durdir $lbldurdir; do
            cp $dir/$class/feats.scp $dir/$class/feats_tmp.scp
            lst=${lst:+$lst,}$dir/$class/feats_tmp.scp
        done
        for dir in $acdir $lbldir $pitchdir $lblpitchdir $durdir $lbldurdir; do
            cat $dir/$class/feats_tmp.scp | awk -v lst=$lst  '
BEGIN{ nv=split(lst, v, ","); 
  for (i = 1; i <= nv; i++) while (getline < v[i]) {nt[$1] = 1; nk[i "_" $1] = 1;} 
  for (k in nt) {
     add = 1;
     for (i = 1; i <= nv; i++) if (nk[i "_" k] != 1) add=0;
     if (add) n[k]=1
  }
}{
   if (n[$1]) print
}' > $dir/$class/feats.scp
        done
    done

    ##############################
    ## 4. Train DNN
    ##############################

    echo -e "\n\n##### Step 5: training DNNs #####\n"

    echo " ### Step 5a: duration model DNN ###"
    # A. Small one for duration modelling
    rm -rf $dnndurdir
    $cuda_cmd $dnndurdir/_train_nnet.log steps/train_nnet_basic.sh --config conf/dur-nn-splice5.conf \
        $lbldurdir/train $lbldurdir/dev $durdir/train $durdir/dev $dnndurdir || exit 1;

    echo " ### Step 5b: pitch prediction DNN ###"
    rm -rf $dnnf0dir
    $cuda_cmd $dnnf0dir/_train_nnet.log steps/train_nnet_basic.sh --config conf/pitch-nn-splice5.conf \
        $lbldir/train $lbldir/dev $pitchdir/train $pitchdir/dev $dnnf0dir || exit 1;

    # C. Larger DNN for filter acoustic features
    echo " ### Step 5c: acoustic model DNN ###"
    rm -rf $dnndir
    $cuda_cmd $dnndir/_train_nnet.log steps/train_nnet_basic.sh --config conf/full-nn-splice5.conf \
        $lblpitchdir/train $lblpitchdir/dev $acdir/train $acdir/dev $dnndir || exit 1;

    echo " ### Step 5d: fake DNN for comparisons ###"
    rm -rf $dnnffdir
    $cuda_cmd $dnnffdir/_train_nnet.log steps/train_nnet_basic.sh --config conf/full-nn-splice5.conf \
        $lbldir/train $lbldir/dev $acdir/train $acdir/dev $dnnffdir || exit 1;

    incr_stage
fi

##############################
## 5. Synthesis
##############################

if [ "$srate" = "16000" ]; then
    order=39
    alpha=0.42
    fftlen=1024
    bndap_order=21
elif [ "$srate" = "48000" ]; then
    order=60
    alpha=0.55
    fftlen=4096
    bndap_order=25
fi

if [ $stage -le 6 ]; then

    # Variant with mlpg: requires mean / variance from coefficients
    copy-feats scp:data/train/feats.scp ark:- \
        | add-deltas --delta-order=2 ark:- ark:- \
        | compute-cmvn-stats --binary=false ark:- - \
        | awk '
(NR==2){count=$NF; for (i=1; i < NF; i++) mean[i] = $i / count}
(NR==3){if ($NF == "]") NF -= 1; for (i=1; i < NF; i++) var[i] = $i / count - mean[i] * mean[i]; nv = NF-1}
END{for (i = 1; i <= nv; i++) print mean[i], var[i]}' \
    > data/train/var_cmp.txt

    # Variant with mlpg: requires mean / variance from coefficients
    copy-feats scp:data/train/pitch_feats.scp ark:- \
        | add-deltas --delta-order=2 ark:- ark:- \
        | compute-cmvn-stats --binary=false ark:- - \
        | awk '
(NR==2){count=$NF; for (i=1; i < NF; i++) mean[i] = $i / count}
(NR==3){if ($NF == "]") NF -= 1; for (i=1; i < NF; i++) var[i] = $i / count - mean[i] * mean[i]; nv = NF-1}
END{for (i = 1; i <= nv; i++) print mean[i], var[i]}' \
    > data/train/var_pitch.txt

    echo -e "\n\n#### Step 6: packaging DNN voice ####\n"

    utils/make_dnn_voice_pitch.sh --spk $spk --srate $srate --mcep_order $order --bndap_order $bndap_order --alpha $alpha --fftlen $fftlen --durdnndir $dnndurdir --f0dnndir $dnnf0dir --acsdnndir $dnndir 

    echo "Voice packaged successfully. Portable models have been stored in ${spk}_pmdl."
    echo "Synthesis can be performed using: 
         echo \"This is a demo of D N N synthesis\" | utils/synthesis_voice_pitch.sh ${spk}_pmdl <outdir>"

    incr_stage
fi
