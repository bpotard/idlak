#!/bin/sh
single_opts=
if [ $1 == "--single" ]; then
    single_opts=1
    shift
fi

exp=$1
tst=$2
out=$3
fmllr=$4
nnet=$5
# we have to reverse the fmllr transformation for speaker $spk
#out=$exp/tst_forward_fmllr
cmpdir=$out/cmp/
wavdir=$out/wav/
mkdir -p $cmpdir
mkdir -p $wavdir
infeats_tst="ark:copy-feats scp:$tst/feats.scp ark:- |"

# input options: only feature transform and global cmvn supported
if [ -e $exp/input_final.feature_transform ]; then
    echo "Applying feature transform on labels"
    infeats_tst="$infeats_tst nnet-forward $exp/input_final.feature_transform ark:- ark:- |"
fi 
if [ -e $exp/indelta_order ]; then
    indelta_order=`cat < $exp/indelta_order`
    echo "Applying deltas order $indelta_order on labels"
    infeats_tst="$infeats_tst add-deltas --delta-order=$indelta_order ark:- ark:- |"
fi
if [ -e $exp/cmvn_glob.ark ]; then
    echo "Applying global cmvn on labels"
    infeats_tst="$infeats_tst apply-cmvn --print-args=false --norm-vars=true $exp/cmvn_glob.ark ark:- ark:- |"
fi
if [ -e $exp/reverse_final.feature_transform ]; then
    feat_transf=$exp/reverse_final.feature_transform
else
    cat $exp/final.feature_transform | convert_transform.sh > $out/reverse_final.feature_transform
    feat_transf=$out/reverse_final.feature_transform
fi
postproc="ark,t:| cat "

# optionally add fmllr transform
if [ "$fmllr" != "" ]; then
    echo "Applying (reversed) fmllr transformation per-speaker"
    postproc="$postproc | reverse-transform-feats --utt2spk=ark:$tst/utt2spk ark:$fmllr ark:- ark,t:-"
fi
# HACK! To remove once deltas are puyt in a reasonable place
#postproc="$postproc | select-feats 0-62 ark:- ark,t:-"
#optionally add cmvn
if [ -e $exp/norm_vars ]; then
    echo "Applying (reversed) per-speaker cmvn on output features"
    norm_vars=`cat < $exp/norm_vars`
    postproc="$postproc | apply-cmvn --reverse --norm-vars=false --utt2spk=ark:$tst/utt2spk scp:${tst/lbldata/data}/cmvn.scp ark:- ark,t:-"
fi

# optionally add global cmvn applied on output
if [ \( "$fmllr" == "" \) -a \( -e $exp/cmvn_out_glob.ark \) ]; then
    echo "Applying (reversed) global cmvn on output feature"
    #norm_vars=`cat < $exp/norm_vars`
    postproc="$postproc | apply-cmvn --reverse --norm-vars=true $exp/cmvn_out_glob.ark ark:- ark,t:- "
    #cmvn-to-nnet --binary=false $exp/cmvn_out_glob.ark - | convert_transform.sh > $exp/reverse_cmvn_out_glob.nnet
    #postproc="$postproc | nnet-forward --no-softmax=true $exp/reverse_cmvn_out_glob.nnet ark:- ark,t:- "
fi
if [ "$single_opts" == "1" ]; then
    postproc="$postproc | copy-feats ark:- ark,scp:$out/feats.ark,$out/feats.scp"
else
    awkcmd="'"'($2 == "["){if (out) close(out); out=dir $1 ".cmp";}($2 != "["){if ($NF == "]") $NF=""; print $0 > out}'"'"
    postproc="$postproc | tee $out/feats.ark | awk -v dir=$cmpdir $awkcmd"
fi
#awkcmd="'"'($2 == "["){if (out) close(out); out=dir $1 ".cmp";}($2 != "["){if ($NF == "]") $NF=""; print $0 > out}'"'"
#postproc="$postproc | tee $out/feats.ark | awk -v dir=$cmpdir $awkcmd"

if [ "$nnet" == "" ]; then
    nnet=$exp/final.nnet
fi

echo "${infeats_tst}"
echo "${postproc}"

nnet-forward --reverse-transform=true --feature-transform=$feat_transf $nnet "${infeats_tst}" "${postproc}"

#for i in $cmpdir/*.cmp; do dnt_tools/straight_synthesis189.sh $i $wavdir/`basename $i .cmp`.wav; done
