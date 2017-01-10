// feat/feature-aperiodic.cc

// Copyright 2013  Arnab Ghoshal

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.



#include "feat/feature-aperiodic.h"
#include "feat/mel-computations.h"
#include <complex>

namespace kaldi {

AperiodicEnergy::AperiodicEnergy(const AperiodicEnergyOptions &opts)
    : opts_(opts), feature_window_function_(opts.frame_opts),
      srfft_(NULL), mel_banks_(NULL) {
  int32 window_size = opts.frame_opts.PaddedWindowSize();
  KALDI_LOG << "Window size: " <<  window_size;
  // If the signal window size is N, then the number of points in the FFT
  // computation is the smallest power of 2 that is greater than or equal to 2N
  padded_window_size_ = RoundUpToNearestPowerOfTwo(window_size);
  if (padded_window_size_ <=
      static_cast<int32>(opts_.frame_opts.samp_freq/opts_.f0_min)) {
    KALDI_ERR << "Padded window size (" << padded_window_size_ << ") too small "
              << " to capture F0 range.";
  }
  srfft_ = new SplitRadixRealFft<BaseFloat>(padded_window_size_);

  FrameExtractionOptions mel_frame_opts = opts_.frame_opts;
  mel_frame_opts.frame_length_ms =
      (static_cast<BaseFloat>(padded_window_size_)
          * 1000.0) / mel_frame_opts.samp_freq;
  if (opts_.use_hts_bands)
      opts_.mel_opts.use_bark_scale = true;
  mel_banks_ = new MelBanks(opts_.mel_opts, mel_frame_opts, 1.0);
}

AperiodicEnergy::~AperiodicEnergy() {
  if (srfft_ == NULL) {
    KALDI_WARN << "NULL srfft_ pointer: This should not happen if the class "
               << "was used properly";
    return;
  }
  delete srfft_;
  if (mel_banks_ == NULL) {
    KALDI_WARN << "NULL mel_banks_ pointer: This should not happen if the class "
               << "was used properly";
    return;
  }
  delete mel_banks_;
}

#define bark(x) (1960.0 / (26.81 / ((x) + 0.53) - 1))

void AperiodicEnergy::Compute_HTS_bands(const VectorBase<BaseFloat> &power_spectrum, Vector<BaseFloat> *output) {
    output->SetZero();
    BaseFloat srate = opts_.frame_opts.samp_freq;
    int32 nbBands = output->Dim();
    
    int32 fftlen = 2 * (power_spectrum.Dim() - 1), bs = 0, be;
    for (int b = 0; b < nbBands; b++) {
	// Legacy hard-coded 5 bands from straight
	if (nbBands == 5) {
	    switch(b) {
	    case 0:
		be = fftlen / 16;
		break;
	    case 1:
		be = fftlen / 8;
		break;
	    case 2:
		be = fftlen / 4;
		break;
	    case 3:
		be = fftlen * 3 / 8;
		break;
	    case 4:
		be = fftlen / 2 + 1;
		break;
	    }
	} else { 	// use buggy hardcoded Bark-inspired bands from Junishi
                        // Note that we use bands that are less buggy, i.e. their size is strictly increasing
			// We fixed other bugs:
	                //  - if bands higher than Nyqist are requested, they are set to 0
                        //  - last band end always set to Nyqist frequency
	    BaseFloat startf, endf; 
	    startf = bark(b);
	    endf = bark(b+1);
	    if (b == 0) startf = 0.0;
	    bs =  static_cast<int32>(round((startf) * fftlen / srate));
	    // Above nyqist frequency or last band?
	    if (endf >= srate / 2 || b >= 25 || b == nbBands - 1) {
		endf = srate / 2;
		be = fftlen / 2 + 1;
	    }
	    else
		be = static_cast<int32>(round((endf) * fftlen / srate));
	    /*KALDI_LOG << "band " << b << " [" << startf << ":" << endf << "] -> " << (be - bs);*/
	}
	// That should never happen, but just in case...
	if (be <= bs) continue;
	float sum = 0.0;
	// NB: in HTS, they normally do geometric mean instead of arithmetic mean; this
	// create risks of underflow (and is probably idiotic anyway) so we do not reproduce
	// that behaviour
	for (int32 i = bs; i < be; i++) {
	    sum += power_spectrum(i);
	}
	(*output)(b) = sum / (be - bs);
	bs = be;
	if (bs >= fftlen / 2 + 1) break;
    }
}

// Note that right now the number of frames in f0 and voicing_prob may be
// different from what Kaldi extracts (due to different windowing in ESPS
// get_f0). There is code to ignore some trailing frames if the difference
// is not large, but those code should be removed once we have a native Kaldi
// F0 extractor.
void AperiodicEnergy::Compute(const VectorBase<BaseFloat> &wave,
                              const VectorBase<BaseFloat> &voicing_prob,
                              const VectorBase<BaseFloat> &f0,
                              Matrix<BaseFloat> *output,
                              Vector<BaseFloat> *wave_remainder) {
  KALDI_ASSERT(output != NULL);
  KALDI_ASSERT(srfft_ != NULL &&
               "srfft_ must not be NULL if class is initialized properly.");
  KALDI_ASSERT(mel_banks_ != NULL &&
               "mel_banks_ must not be NULL if class is initialized properly.");
  int32 frames_out = NumFrames(wave.Dim(), opts_.frame_opts),
      dim_out = opts_.mel_opts.num_bins;
  if (frames_out == 0)
    KALDI_ERR << "No frames fit in file (#samples is " << wave.Dim() << ")";
  //  if (voicing_prob.Dim() != frames_out) {
  // The following line should be replaced with the line above eventually
  if (std::abs(voicing_prob.Dim() - frames_out) > opts_.frame_diff_tolerance) {
    KALDI_ERR << "#frames in probability of voicing vector ("
              << voicing_prob.Dim() << ") doesn't match #frames in data ("
              << frames_out << ").";
  }
  //  if (f0.Dim() != frames_out) {
  // The following line should be replaced with the line above eventually
  if (std::abs(f0.Dim() - frames_out) > opts_.frame_diff_tolerance) {
    KALDI_ERR << "#frames in F0 vector (" << f0.Dim() << ") doesn't match "
              << "#frames in data (" << frames_out << ").";
  }

  frames_out = std::min(frames_out, f0.Dim());  // will be removed eventually
  output->Resize(frames_out, dim_out);
  if (wave_remainder != NULL)
    ExtractWaveformRemainder(wave, opts_.frame_opts, wave_remainder);
  Vector<BaseFloat> wave_window;
  Vector<BaseFloat> padded_window(padded_window_size_, kUndefined);
  Vector<BaseFloat> binned_energies(dim_out);
  Vector<BaseFloat> noise_binned_energies(dim_out);
  if (!opts_.use_hts_bands || dim_out != 5) {
      Vector<BaseFloat> frqs = mel_banks_->GetCenterFreqs();
      BaseFloat freq_delta;
      /* The center frequencies are equally spaced in the mel / bark
         domain */
      if (opts_.mel_opts.use_bark_scale) 
          freq_delta = mel_banks_->BarkScale(frqs(1)) - mel_banks_->BarkScale(frqs(0));
      else
          freq_delta = mel_banks_->MelScale(frqs(1)) - mel_banks_->MelScale(frqs(0));
      for (int i = 0; i < frqs.Dim(); i++) {
          BaseFloat low_freq, high_freq;
          // simulating rectangular window
          if (opts_.mel_opts.use_bark_scale) {
              low_freq = mel_banks_->InverseBarkScale(mel_banks_->BarkScale(frqs(i)) - 0.5 * freq_delta);
              high_freq = mel_banks_->InverseBarkScale(mel_banks_->BarkScale(frqs(i)) + 0.5 * freq_delta);
          }
          else {
              low_freq = mel_banks_->InverseMelScale(mel_banks_->MelScale(frqs(i)) - 0.5 * freq_delta);
              high_freq = mel_banks_->InverseMelScale(mel_banks_->MelScale(frqs(i)) + 0.5 * freq_delta);
          }
          KALDI_LOG << "Band " << i << ": [ " << low_freq << " ; " << frqs(i) << " ; " << high_freq << " ]";
      }
  }

  for (int32 r = 0; r < frames_out; r++) {  // r is frame index
    SubVector<BaseFloat> this_ap_energy(output->Row(r));
    ExtractWindow(wave, r, opts_.frame_opts, feature_window_function_,
                  &wave_window, NULL);
    int32 window_size = wave_window.Dim();
    padded_window.SetZero();
    padded_window.Range(padded_window_size_/2 - window_size/2, window_size).CopyFromVec(wave_window);
    srfft_->Compute(padded_window.Data(), true);

    Vector<BaseFloat> tmp_spectrum(padded_window);
    ComputePowerSpectrum(&tmp_spectrum);
    SubVector<BaseFloat> power_spectrum(tmp_spectrum, 0,
                                        padded_window_size_/2 + 1);
    KALDI_LOG << "Computing frame " << r << " with F0 " << f0(r); 

    /* BP: this code allows us to speed things up a bit for unvoiced regions
       but it may introduce some jumps for no good reasons, so commented out for now! */
    if (voicing_prob(r) <= 0.0) {  // unvoiced region
      // While the aperiodic energy will not be used during synthesis of the
      // unvoiced regions, it is still modeled as a separate stream in the HMM
      // and so we set it to the log-filterbank values.
      //mel_banks_->Compute(power_spectrum, &binned_energies);
      //binned_energies.ApplyLog();
        for (int i = 0; i < dim_out; i++) binned_energies(i) = 0.0;
        this_ap_energy.CopyFromVec(binned_energies);
        continue;
	}
    if (opts_.use_hts_bands && dim_out == 5)
        Compute_HTS_bands(power_spectrum, &binned_energies);
    else
        mel_banks_->Compute(power_spectrum, &binned_energies);
    

    std::vector<bool> noise_indices;
    IdentifyNoiseRegions(power_spectrum, f0(r), &noise_indices);

    // padded_window contains the FFT coefficients
    ObtainNoiseSpectrum(padded_window, noise_indices, &tmp_spectrum);
    SubVector<BaseFloat> noise_spectrum(tmp_spectrum, 0,
                                        padded_window_size_/2 + 1);
    if (opts_.use_hts_bands && dim_out == 5)
        Compute_HTS_bands(noise_spectrum, &noise_binned_energies);
    else
        mel_banks_->Compute(noise_spectrum, &noise_binned_energies);
  
	
    /* Normalise noise energy by the total energy in the band */
    for (int i = 0; i < noise_binned_energies.Dim(); i++) {
        /*if (i == 0)
          KALDI_LOG << "band " << i << ": " << noise_binned_energies(i) /  binned_energies(i);*/
        noise_binned_energies(i) /= binned_energies(i);
        if (noise_binned_energies(i) >= 1.0) noise_binned_energies(i) = 1.0;
        /*KALDI_LOG << noise_binned_energies(i) << " " <<  binned_energies(i);*/
    }
    noise_binned_energies.ApplyLog();  // take the log
    this_ap_energy.CopyFromVec(noise_binned_energies);
  }
}

//#define VBOOST 1.5

// For voiced regions, we reconstruct the harmonic spectrum from the cepstral
// coefficients around the cepstral peak corresponding to F0, and noise spectrum
// using the rest of the cepstral coefficients. The frequency samples for which
// the noise spectrum has a higher value than the harmonic spectrum are returned.
void AperiodicEnergy::IdentifyNoiseRegions(
    const VectorBase<BaseFloat> &power_spectrum,
    BaseFloat f0,
    std::vector<bool> *noise_indices) {
  KALDI_ASSERT(noise_indices != NULL);
  KALDI_ASSERT(power_spectrum.Dim() == padded_window_size_/2+1 && "Power "
               "spectrum size expected to be half of padded window plus 1.");
  BaseFloat sampling_freq = opts_.frame_opts.samp_freq;
  int32 f0_index = static_cast<int32>(round(sampling_freq/f0)),
      max_f0_index = static_cast<int32>(round(sampling_freq/opts_.f0_max)),
      peak_index, test_index;
  // Width of F0-induced cepstral peak
  int32 peak_width = static_cast<int32>(round(f0_index * opts_.f0_width * 0.5)), npeak_width;
  int32 trans_width = static_cast<int32>(round(max_f0_index * 0.1));
  noise_indices->resize(padded_window_size_/2+1, false);

  Vector<BaseFloat> noise_spectrum(padded_window_size_, kSetZero);
  Vector<BaseFloat> test_spectrum(padded_window_size_, kSetZero);
  Vector<BaseFloat> filter_spectrum(padded_window_size_, kSetZero);
  Vector<BaseFloat> abs_spectrum(padded_window_size_, kSetZero);
  noise_spectrum.Range(0, padded_window_size_/2+1).CopyFromVec(power_spectrum);
  // Hacky: to remove high frequency noise in cepstrum, we do a low-pass filter on power spectrum,
  // only keeping up to 4k
  int32 k4_idx =  static_cast<int32>(round(padded_window_size_ * 4000.0 / sampling_freq));
  for (int32 i = 0; i <= padded_window_size_/2; i++) {
      if (i < trans_width * 3) {
	  noise_spectrum(k4_idx + i) *= (0.501 + 0.499 * cos((i * M_PI / (trans_width * 3)  )));
      }
      else {
	  noise_spectrum(k4_idx + i) *= 0.01;
      }
  }
  // Original spectrum
  if ( opts_.debug_aperiodic) {
      fprintf(stderr, "O");
      for (int i = 0; i <= padded_window_size_/2; ++i) {
	  fprintf(stderr, " %f", log(sqrt(power_spectrum(i))));
      }
      fprintf(stderr, "\n");
  }
  // Compute Real Cepstrum
  PowerSpecToRealCeps(&noise_spectrum);
  // Filter ("vocal tract") spectrum
  if ( opts_.debug_aperiodic) {
      filter_spectrum.CopyFromVec(noise_spectrum);
      filter_spectrum.Range(max_f0_index, padded_window_size_/2).SetZero();
      for (int32 i = -trans_width; i <= 0; i++) {
	  filter_spectrum(max_f0_index + i) *= (0.5 - 0.5 * cos(i * M_PI / trans_width ));
      }
      RealCepsToMagnitudeSpec(&filter_spectrum, false /* get log spectrum*/);
      fprintf(stderr, "F");
      for (int i = 0; i <= padded_window_size_/2; ++i) {
	  fprintf(stderr, " %f", filter_spectrum(i));
      }
      fprintf(stderr, "\n");
  }

  // All cepstral coefficients below the one corresponding to maximum F0 are
  // considered to correspond to vocal tract characteristics and ignored from
  // the initial harmonic to noise ratio calculation.
  noise_spectrum.Range(0, max_f0_index - trans_width- 1).SetZero();
  // Cos lifter for smoother spectrum
  for (int32 i = -trans_width; i <= 0; i++) {
      noise_spectrum(max_f0_index + i) *= (0.5 + 0.5 * cos(i * M_PI / trans_width ));
  }
  /* Find peak in amplitude spectrum */
  for (int32 i = 0; i <=  padded_window_size_/2; i++) {
      abs_spectrum(i) = fabs(noise_spectrum(i));
  }
  abs_spectrum.Max(&peak_index);
  npeak_width = static_cast<int32>(round(peak_index * opts_.f0_width * 0.5));
  // Estimate quality of peak by zeroing peak area and getting max of left-over
  test_spectrum.CopyFromVec(abs_spectrum);
  // Set to 0 from peak_index-npeak_width to peak_index+npeak_width
  // But maybe we should actually set everything above peak_index-npeak_width to 0?
  test_spectrum.Range(peak_index-npeak_width, npeak_width * 2).SetZero();
  test_spectrum.Max(&test_index);
  // Quality between 0.0 (very bad) and 1.0 (perfect)
  // In unvoiced region the peak should be less than 0.1, in well voiced region above 0.5
  BaseFloat peak_quality = 1.0 - 1.0 / noise_spectrum(peak_index) * test_spectrum(test_index);

  if (peak_index < f0_index-peak_width || peak_index > f0_index+peak_width) {
      KALDI_VLOG(2) << "Actual cepstral peak (index=" << peak_index << "; value = "
              << abs_spectrum(peak_index) << "; quality = "
	      << peak_quality << ") occurs too far from F0 (index="
              << f0_index << "; value = " << noise_spectrum(f0_index) << ").";
    //f0_index = peak_index;  // TODO(arnab): remove this: it is only for testing.
  }
  // Slightly hacky, a good quality peak will replace F0
  if (f0 == 0 || ((peak_quality >= opts_.quality_threshold) && (f0_index != peak_index))) {
      KALDI_VLOG(2) << "Actual cepstral peak has quality "<< peak_quality << ", moving f0 from "<< f0_index << " to " << peak_index;
      f0_index = peak_index;
      peak_width = npeak_width;
  }
  // Print cepstral coefficients
  if ( opts_.debug_aperiodic) {
      fprintf(stderr, "C");
      for (int i = 0; i <= padded_window_size_/2; ++i) {
	  fprintf(stderr, " %f", noise_spectrum(i));
      }
      fprintf(stderr, "\n");
  }
  
  // Generating harmonic spectrum from cepstral coefficients located
  // around cepstral / f0 peak
  Vector<BaseFloat> harmonic_spectrum(padded_window_size_, kSetZero);
  // Note that at this point noise_spectrum contains cepstral coeffs
  // energy
  // In the range of interest, we copy the ceps to harmonic_spectrum
  // Note that to have the "real" noise spectrum, we should zero out these
  // same quefrencies.
  for (int32 k = 1; k <= 4; k++) {
      if (k * f0_index + peak_width >= padded_window_size_/2) break;
      // lifter before and after the peak
      int32 offset = k * f0_index;
      /*for (int32 i = - trans_width; i < 0; i++) {
	  BaseFloat m = VBOOST * (0.5 + 0.5 * cos(i * M_PI / trans_width ));
	  // Before peak
	  BaseFloat val = noise_spectrum(offset - peak_width + i);
	  harmonic_spectrum(offset - peak_width + i) = m * val;
	  // After peak
	  val = noise_spectrum(offset + peak_width - i);
	  harmonic_spectrum(offset + peak_width - i) = m * val;
	  }*/
      // copy within the peak
      for (int32 i = - peak_width; i <= peak_width; i++) {
	  BaseFloat m = (0.5 + 0.5 * cos(i * M_PI / (peak_width + 1) ));
	  harmonic_spectrum(offset + i) = m * noise_spectrum(offset + i);
	  /*noise_spectrum(offset + i) *= (1 - m);*/
      }
  }
  RealCepsToMagnitudeSpec(&harmonic_spectrum, false /* get log spectrum*/);
  RealCepsToMagnitudeSpec(&noise_spectrum, false /* get log spectrum*/);
  // Print Harmonic and Harmonic + Noise spectrograms
  if ( opts_.debug_aperiodic) {
      fprintf(stderr, "H");
      for (int i = 0; i <= padded_window_size_/2; ++i) {
	  fprintf(stderr, " %f", harmonic_spectrum(i));
      }
      fprintf(stderr, "\nN");
      for (int i = 0; i <= padded_window_size_/2; ++i) {
	  fprintf(stderr, " %f", noise_spectrum(i));
      }
      fprintf(stderr, "\n");
  }

  // We find the "negative peaks" of the harmonic spectrum, we consider these will
  // always correspond to noise spectrum frequencies. We will then interpolate between these
  // point to build the full "noise spectrogram".
  // This will work well in voiced regions if the F0 has been estimated correctly. 
  // This will work pretty well in noise regions as well, unless somehow the "noise"
  // has strong harmonics :-)
  // Our peak picking is a disgrace, but it works well enough!
#define _is_neg_peak(s, i) ((s(i) < s(i-1)) && (s(i) < s(i+1)) && (s(i) < 0.0)) 
  // NB: the H+N spectrogram is not used at all anymore.
  for (int32 i = 0; i <= padded_window_size_/2; ++i) {
      // NB: to behave like STRAIGHT, it seems you should just use "if ( harmonic_spectrum(i) < 0)"
      // in 0, the aperiodic energy should always be 0.0!
      if (i > 0 && i < padded_window_size_/2 && _is_neg_peak(harmonic_spectrum, i)) {
	  (*noise_indices)[i] = true;
      }
      /* Hacky bit that enforces the first negative dip in the original spectrum to be marked as noise;
	 not sure it is needed as long as we enforce the "0" frequency to be harmonic. */
      /*else if (first_peak && i > 0 && noise_spectrum(i) <= noise_spectrum(i-1) && noise_spectrum(i) <= noise_spectrum(i+1)) {
	  (*noise_indices)[i] = true;
	  first_peak = false;
	  }*/
      else (*noise_indices)[i] = false;
  }
}

void AperiodicEnergy::ObtainNoiseSpectrum(
    const VectorBase<BaseFloat> &fft_coeffs,
    const std::vector<bool> &noise_indices,
    Vector<BaseFloat> *noise_spectrum) {
  KALDI_ASSERT(noise_spectrum != NULL);
  KALDI_ASSERT(noise_spectrum->Dim() == padded_window_size_);

  noise_spectrum->SetZero();
  Vector<BaseFloat> prev_estimate(padded_window_size_, kSetZero);
  Vector<BaseFloat> debug_spectrum(padded_window_size_, kSetZero);
  Vector<BaseFloat> window_array(padded_window_size_, kSetZero);
  int32 window_size = opts_.frame_opts.WindowSize();
  int32 last_ind = 0;
  window_array.Range(padded_window_size_/2 - window_size/2, window_size).CopyFromVec(feature_window_function_.window);
  for (int32 iter = 0; iter < opts_.max_iters; ++iter) {
    // Copy the DFT coefficients for the noise-regions
      last_ind = -1;
      // NB: frequency 0 should never be noise. But we don't enforce it :-)
    for (int32 j = 0; j <= padded_window_size_/2; ++j) {
	if ((j == padded_window_size_/2) || noise_indices[j]) {
	    if (j != padded_window_size_/2) {
		(*noise_spectrum)(j*2) = fft_coeffs(j*2);
		(*noise_spectrum)(j*2+1) = fft_coeffs(j*2+1);
	    }
	    // Some gaps have to be filled in
	    if ((opts_.max_iters == 1) && (last_ind != j - 1)) {
		// interpolate linearly between last_ind and j-1
		if (last_ind == -1) last_ind = 0;
		BaseFloat init_re, init_im, end_re, end_im;
		/* Handles Kaldi idiotic FFT structure: 0 and nyqist frequencies real together, 
		   all the rest as complex */
		// Start from 0
		if (last_ind == 0) {
		    init_re = (*noise_spectrum)(0);
		    init_im = 0.0;
		} else {
		    init_re = fft_coeffs(last_ind * 2);
		    init_im = fft_coeffs(last_ind * 2 + 1);
		}
		// End at Nyqist, always noise
		if (j == padded_window_size_/2) {
		    end_re = fft_coeffs(1);
		    end_im = 0.0;
		} else {
		    end_re = fft_coeffs(j * 2);
		    end_im = fft_coeffs(j * 2 + 1);
		}
		std::complex<double> init_cplx(init_re, init_im), end_cplx(end_re, end_im);
		double init_mod = std::abs(init_cplx), end_mod = std::abs(end_cplx);
		double init_arg = std::arg(init_cplx), end_arg = std::arg(end_cplx);
		
		for (int k = last_ind + 1; k < j; k++) {
		    float m = ((float)k - last_ind) / (j - last_ind);
		    std::complex<double> z = std::polar(init_mod * (1.0 - m) + end_mod * m, init_arg * (1.0 - m) + end_arg * m);
		    (*noise_spectrum)(k*2) = (BaseFloat)z.real();
		    (*noise_spectrum)(k*2 + 1) = (BaseFloat)z.imag();
		}
	    }
	    last_ind = j;
	}
    }
    // Interpolate between noise bands */
    if (opts_.max_iters == 1) {
	break;
    }
    /*if (iter % 10 == 0) {
	debug_spectrum.CopyFromVec(*noise_spectrum);
	ComputePowerSpectrum(&debug_spectrum);
	fprintf(stderr, "NI%d", iter);
	for (int i = 0; i <= padded_window_size_/2; ++i) {
	    fprintf(stderr, " %f", debug_spectrum(i));
	}
	fprintf(stderr, "\n");
	}*/
    srfft_->Compute(noise_spectrum->Data(), false /*do IFFT*/);
    BaseFloat ifft_scale = 1.0/padded_window_size_;
    noise_spectrum->Scale(ifft_scale);
    // Zero out most of the frame, except for window_size elements
    noise_spectrum->Range(window_size, padded_window_size_-window_size).SetZero();
    // Should pretty much do what we want, avoiding high freq noise
    /*noise_spectrum->MulElements(window_array);*/
    /*noise_spectrum->Range(0, padded_window_size_/4).SetZero();
      noise_spectrum->Range(3 * padded_window_size_/4, padded_window_size_/4).SetZero();*/
    //noise_spectrum->Range(padded_window_size_/2 + window_size / 2, padded_window_size_/2 - window_size / 2).SetZero();
    /*if (iter > 0) {  // calculate the squared error (in time domain)
      prev_estimate.AddVec(-1.0, *noise_spectrum);
      BaseFloat err = prev_estimate.SumPower(2.0) / ifft_scale / 32768.0;
      //KALDI_LOG << "Iteration " << iter
      //          << ": Aperiodic component squared error = " << err;
      if (err < opts_.min_sq_error) {  // converged
        // noise_spectrum is still in time domain; convert to frequency domain
        srfft_->Compute(noise_spectrum->Data(), true //do FFT);
        break;
      }
    }*/
    prev_estimate.CopyFromVec(*noise_spectrum);

    srfft_->Compute(noise_spectrum->Data(), true /*do FFT*/);
  }
  ComputePowerSpectrum(noise_spectrum);
  if ( opts_.debug_aperiodic) {
      fprintf(stderr, "B");
      for (int i = 0; i <= padded_window_size_/2; ++i) {
	  fprintf(stderr, " %f", log(sqrt((*noise_spectrum)(i))));
      }
      fprintf(stderr, "\n");
  }
}

}  // namespace kaldi
