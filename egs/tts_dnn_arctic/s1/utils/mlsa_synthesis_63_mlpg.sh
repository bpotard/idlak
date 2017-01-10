#!/bin/bash
#
# Copyright 2013 by Idiap Research Institute, http://www.idiap.ch
#
# See the file COPYING for the licence associated with this software.
#
# Author(s):
#   Blaise Potard, 2013
#

# Allow setshell
#software=/idiap/resource/software
#source $software/initfiles/shrc $software
#win=/remote/lustre/1/temp/bpotard/src/kaldi-trunk/egs/TTS/win

# Perform raw resynthesis of cmp file
#SETSHELL straight
#SETSHELL sptk

#TMP=/scratch/$USER/tmp
#mkdir -p $TMP
synth=excitation # or excitation, cere
period=5
srate=16000
delta_order=0
mcep_order=39
bndap_order=21
f0_order=2
# Continuous f0, no mixing
voice_thresh=0.8
alpha=0.55
fftlen=1024
tmpdir=/tmp
win=win

[ -f path.sh ] && . ./path.sh; 
. parse_options.sh || exit 1;


cmp_file=$1
out_wav=$2
var_file=$3
base=`basename $cmp_file .cmp`
var=$tmpdir/var
cmp=$tmpdir/$base.cmp
uv=$tmpdir/$base.uv
#variance file
vmcep=$tmpdir/$base.mcep.var
vf0=$tmpdir/$base.f0.var
vbap=$tmpdir/$base.bndap.var
#pdf files
mpdf=$tmpdir/$base.mcep.pdf
fpdf=$tmpdir/$base.f0.pdf
bpdf=$tmpdir/$base.bndap.pdf
#feature files
mcep=$tmpdir/$base.mcep
f0=$tmpdir/$base.f0
bap=$tmpdir/$base.bndap

apf=$tmpdir/$base.apf
sspec=$tmpdir/$base.spec


#mcep_order=39
# Assuming: F0 - MCEP - BNDAP
delta_mult=$(( $delta_order + 1 ))
f0_offset=1
f0_len=$(( $f0_order * $delta_mult ))

mcep_offset=$(( $f0_offset + $f0_len ))
mcep_len=$(( ($mcep_order + 1) * $delta_mult ))
order=$mcep_order

bndap_offset=$(( $mcep_offset + $mcep_len ))
bndap_len=$(( $bndap_order * $delta_mult ))

echo $f0_offset $f0_len $mcep_offset $mcep_len $bndap_offset $bndap_len
cat $cmp_file | sed -e 's/^ *//g' -e 's/ *$//g' \
    | awk -v nit=$delta_mult \
    -v off=$(( $f0_order + $mcep_order + 1 + $bndap_order )) \
    -v bnd="$f0_offset;$(($f0_offset + $f0_order));$(($f0_offset + $f0_order + $mcep_order + 1))" \
'
BEGIN{nb = split(bnd, bnda, ";"); bnda[nb + 1] = off+1; nb += 1;}
{ 
  for (b = 1; b < nb; b++) {
    for (k = 0; k < nit; k++) {
      for (i = bnda[b]; i < bnda[b+1]; i++) {
         printf "%f ", $(i + k * off); 
      }
    }
  }
  printf "\n"
}' > $cmp


if [ "$var_file" != "" ]; then
    if [ $delta_mult -eq 2 ]; then
	mcep_win="-d $win/mcep_d1.win"
	f0_win="-d $win/logF0_d1.win"
	bndap_win="-d $win/bndap_d1.win"
    elif [ $delta_mult -eq 3 ]; then
	mcep_win="-d $win/mcep_d1.win -d $win/mcep_d2.win"
	f0_win="-d $win/logF0_d1.win -d $win/logF0_d2.win"
	bndap_win="-d $win/bndap_d1.win -d $win/bndap_d2.win"
    else
	mcep_win=""
	f0_win=""
	bndap_win=""
    fi
    echo "Extracting variances..."
    cat $var_file | awk '{printf "%f ", $2}' > $var
    cat $var | cut -d " " -f $mcep_offset-$(( $mcep_offset + $mcep_len - 1 )) > $vmcep
    cat $var | cut -d " " -f $f0_offset-$(( $f0_offset + $f0_len - 1 )) > $vf0
    cat $var | cut -d " " -f $bndap_offset-$(( $bndap_offset + $bndap_len - 1 )) > $vbap

    echo "Creating pdfs..."
    cat $cmp | cut -d " " -f $mcep_offset-$(( $mcep_offset + $mcep_len - 1 )) | awk -v var="`cat $vmcep`" '{print $0, var}' | x2x +a +f > $mpdf
    cat $cmp | cut -d " " -f $f0_offset-$(( $f0_offset + $f0_len - 1 )) | awk -v var="`cat $vf0`" '{print $0, var}' | x2x +a +f > $fpdf
    cat $cmp | cut -d " " -f $bndap_offset-$(( $bndap_offset + $bndap_len - 1 )) | awk -v var="`cat $vbap`" '{print $0, var}' | x2x +a +f > $bpdf
    
    echo "Running mlpg..."
    echo "mcep smoothing"
    mlpg -i 0 -m $mcep_order $mcep_win $mpdf | x2x +f +a$(( $mcep_order + 1 )) > $mcep
    echo "f0 smoothing"
    mlpg -i 0 -m $(( $f0_order - 1 )) $f0_win $fpdf > ${f0}_raw
    echo "bndap smoothing"
    mlpg -i 0 -m $(( $bndap_order - 1 )) $bndap_win $bpdf | x2x +f +a$bndap_order | awk '{for (i = 1; i <= NF; i++) if ($i >= -0.5) printf "0.0 " ; else printf "%f ", 20 * ($i + 0.5) / log(10); printf "\n"}' > $bap

    #cat $cmp | cut -d " " -f $(($f0_offset + 1))-$(($f0_offset + 1)) > $uv
    cat ${f0}_raw | x2x +f +a$f0_order | awk -v thresh=$voice_thresh '{if ($1 > thresh) print $2; else print 0.0}' > $f0
    
    # Do not do mlpg on MCEP
    #cat $cmp | cut -d " " -f $mcep_offset-$(($mcep_offset + $mcep_order)) > $mcep
else
    cat $cmp | cut -d " " -f $mcep_offset-$(($mcep_offset + $mcep_order)) > $mcep
    cat $cmp | cut -d " " -f $f0_offset-$(($f0_offset + $f0_order - 1)) | awk -v thresh=$voice_thresh '{if ($1 > thresh) print $2; else print 0.0}' > $f0
    cat $cmp | cut -d " " -f $bndap_offset-$(($bndap_offset + $bndap_order - 1)) | awk '{for (i = 1; i <= NF; i++) if ($i >= 0.0) printf "0.0 " ; else printf "%f ", 20 * ($i) / log(10); printf "\n"}' > $bap
fi

# if we want to keep the original f0/apf:
#cat $cmp_file | sed -e 's/^ *//g' -e 's/ *$//g' | cut -d " " -f $f0_offset-$(($f0_offset + 1)) | awk -v thresh=$voice_thresh '{if ($2 > thresh) print exp($1); else print 0.0}' > $f0
#cat $cmp_file | sed -e 's/^ *//g' -e 's/ *$//g' | cut -d " " -f $bndap_offset-$(($bndap_offset + $bndap_order - 1)) | x2x +a +f > $bap

#echo "synthesis_fft -mel -bap -order $mcep_order -apfile $bap $f0 $mcep $2"
#/idiap/user/pehonnet/HTS-ENGINE-for-HTS-2.1/
#synthesis_fft -float -f $samp_freq -sigp 1.2 -cornf 1000 -bw 70.0 -delfrac 0.2 -sd 0.5 -mel -bap -order $mcep_order -apfile $bap -alpha $alpha $f0 $mcep $2

if [ "$synth" = "cere" ]; then
    if [ `awk -v vt=$voice_thresh 'BEGIN{if (vt > 0) print 1; else print 0}'` -eq 1 ]; then
        mlopts="-n"
    else
        mlopts="-p"
    fi
    cmd="python $HOME/cereproc/trunk/apps/dsplab/mlsa.py $mlopts -C -a $alpha -m $order -s $srate -f $fftlen -b $bndap_order -i $tmpdir -o $tmpdir $base"
    echo $cmd
    $cmd
    cp $tmpdir/$base.wav $out_wav
elif [ "$synth" = "excitation" ]; then
    x2x +af $mcep > $mcep.float
    psize=`echo "$period * $srate / 1000" | bc`
    # We have to drop the first few F0 frames to match SPTK behaviour
    #cat $f0 | awk -v srate=$srate '(NR > 2){if ($1 > 0) print srate / $1; else print 0.0}' | x2x +af \
    #    | excite -p $psize \
    python utils/excitation.py -G 0.8 -s $srate -f $fftlen -b $bndap_order $f0 $bap > $tmpdir/resid.float
    cat $tmpdir/resid.float | mlsadf -P 5 -m $order -a $alpha -p $psize $mcep.float | x2x -o +fs > $tmpdir/data.mcep.syn
    sox --norm -t raw -c 1 -r $srate -e signed-integer -b 16 $tmpdir/data.mcep.syn $out_wav
else
    x2x +af $mcep > $mcep.float
    psize=`echo "$period * $srate / 1000" | bc`
    # We have to drop the first few F0 frames to match SPTK behaviour
    cat $f0 | awk -v srate=$srate '(NR > 2){if ($1 > 0) print srate / $1; else print 0.0}' | x2x +af \
        | excite -p $psize \
        | mlsadf -P 5 -m $order -a $alpha -p $psize $mcep.float | x2x -o +fs > $tmpdir/data.mcep.syn
    sox --norm -t raw -c 1 -r $srate -e signed-integer -b 16 $tmpdir/data.mcep.syn $out_wav
fi

#-sigp 1.2
#          -sd 0.5
#          -cornf 100
#          -bw 70.0
#          -delfrac 0.2
# rm -f $mcep $f0 $bap $apf $sspec
#synthesis_fft -f $samp_freq -mel -order $mcep_order $f0 $mcep $2
#synthesis_fft -mel -bap -order 40 -float $f0 $mcep /tmp/out.wav -apfile $apf
