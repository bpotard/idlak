// nnetbin/nnet-train-mse-tgtmat-frmshuff.cc

// Copyright 2012-2013  Brno University of Technology (Author: Karel Vesely)

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

/*
 * Karel : This tool has never been used, so there may be bugs.
 */

#include "nnet/nnet-trnopts.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "nnet/nnet-randomizer.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"
#include "cudamatrix/cu-device.h"


int main(int argc, char *argv[]) {
  using namespace kaldi;
  using namespace kaldi::nnet1;
  try {
    const char *usage =
        "Perform one iteration of Neural Network training by stochastic gradient descent.\n"
        "Usage:  nnet-train-mse-tgtmat-frmshuff [options] <model-in> <feature-rspecifier> <targets-rspecifier> [<model-out>]\n"
        "e.g.: \n"
        " nnet-train-mse-tgtmat-frmshuff nnet.init scp:train.scp ark:targets.scp nnet.iter1\n";

    ParseOptions po(usage);

    NnetTrainOptions trn_opts;
    trn_opts.Register(&po);
    NnetDataRandomizerOptions rnd_opts;
    rnd_opts.Register(&po);

    bool binary = false, 
         crossvalidate = false,
         randomize = true;
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("cross-validate", &crossvalidate, "Perform cross-validation (don't backpropagate)");
    po.Register("randomize", &randomize, "Perform the frame-level shuffling within the Cache::");

    std::string feature_transform;
    po.Register("feature-transform", &feature_transform, "Feature transform in Nnet format");

    std::string frame_weights;
    po.Register("frame-weights", &frame_weights, "Per-frame weights to scale gradients (frame selection/weighting).");

    double dropout_retention = 0.0;
    po.Register("dropout-retention", &dropout_retention, "number between 0..1, saying how many neurons to preserve (0.0 will keep original value");

    //int32 bunchsize=512, cachesize=32768, seed=777;
    //po.Register("bunchsize", &bunchsize, "Size of weight update block");
    //po.Register("cachesize", &cachesize, "Size of cache for frame level shuffling (max 8388479)");
    //po.Register("seed", &seed, "Seed value for srand, sets fixed order of frame-shuffling");

#if HAVE_CUDA==1
    std::string use_gpu="yes";
    po.Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA"); 
    //int32 use_gpu_id=-2;
    //po.Register("use-gpu-id", &use_gpu_id, "Manually select GPU by its ID (-2 automatic selection, -1 disable GPU, 0..N select GPU)");
#endif
    
    po.Read(argc, argv);

    if (po.NumArgs() != 4-(crossvalidate?1:0)) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        targets_rspecifier = po.GetArg(3);
        
    std::string target_model_filename;
    if (!crossvalidate) {
      target_model_filename = po.GetArg(4);
    }

    //set the seed to the pre-defined value
    //srand(seed);
     
    using namespace kaldi;
    using namespace kaldi::nnet1;
    typedef kaldi::int32 int32;

    //Select the GPU
#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    Nnet nnet_transf;
    if(feature_transform != "") {
      nnet_transf.Read(feature_transform);
    }

    Nnet nnet;
    nnet.Read(model_filename);
    nnet.SetTrainOptions(trn_opts);
    if (dropout_retention > 0.0) {
      nnet_transf.SetDropoutRetention(dropout_retention);
      nnet.SetDropoutRetention(dropout_retention);
    }

    kaldi::int64 total_frames = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    SequentialBaseFloatMatrixReader targets_reader(targets_rspecifier);
    SequentialBaseFloatVectorReader weights_reader;
    if (frame_weights != "") {
      weights_reader.Open(frame_weights);
    }

    RandomizerMask randomizer_mask(rnd_opts);
    MatrixRandomizer feature_randomizer(rnd_opts);
    MatrixRandomizer targets_randomizer(rnd_opts);
    VectorRandomizer weights_randomizer(rnd_opts);

    //CacheTgtMat cache;
    //cachesize = (cachesize/bunchsize)*bunchsize; // ensure divisibility
    //cache.Init(cachesize, bunchsize);

    Mse mse;

    
    CuMatrix<BaseFloat> feats, feats_transf, targets, nnet_in, nnet_out, nnet_tgt, obj_diff;

    Matrix<BaseFloat> tfeats;

    Timer time;
    double time_now = 0;
    double time_next = 0;
    KALDI_LOG << (crossvalidate?"CROSSVALIDATE":"TRAINING") << " STARTED";

    int32 num_done = 0, num_no_tgt_mat = 0, num_other_error = 0;//, num_cache = 0;
    while (1) {
#if HAVE_CUDA==1
      // check the GPU is not overheated
      CuDevice::Instantiate().CheckGpuHealth();
#endif
      // fill the cache, 
      // both reader are sequential, not to be too memory hungry,
      // the scp lists must be in the same order 
      // we run the loop over targets, skipping features with no targets
      while (!feature_randomizer.IsFull() && !feature_reader.Done()) {
        // get the utts
	//std::string tgt_utt = targets_reader.Key();
	std::string fea_utt = feature_reader.Key();
	std::string tgt_utt = targets_reader.Key();
	while (tgt_utt != fea_utt) {
	  if (targets_reader.Done()) {
	    targets_reader.Close();
	    break;
	  }
	  targets_reader.Next();
	  tgt_utt = targets_reader.Key();
	  // check we have per-frame weights
	  if (frame_weights != "") {
	    weights_reader.Next();
	  }
	}
	if (tgt_utt != fea_utt) {//!targets_reader.HasKey(fea_utt)) {
	  KALDI_WARN << "No targets for: " << fea_utt;
          num_no_tgt_mat++;
	  targets_reader.Open(targets_rspecifier);
	  //targets_reader.Reset();
	}
	else {
          // get feature tgt_mat pair
          const Matrix<BaseFloat> &fea_mat = feature_reader.Value();
	  const Matrix<BaseFloat> &tgt_mat = targets_reader.Value();//fea_utt);
	  Matrix<BaseFloat>  bob;
          // chech for dimension
	  if (tgt_mat.NumRows() != fea_mat.NumRows()) {
	    KALDI_WARN << "Alignment has wrong size "<< (tgt_mat.NumRows()) << " vs. "<< (fea_mat.NumRows()) << " for utterance " << fea_utt;
	    num_other_error++;
            feature_reader.Next();
            targets_reader.Next();
	    continue;
	  }
	  /*else {*/
	    // push features/targets to GPU
	  //KALDI_WARN << "Alignment has size "<< (tgt_mat.NumRows()) << "," << (tgt_mat.NumCols()) << " vs. "<< (fea_mat.NumRows()) << "," << (fea_mat.NumCols());
	  tfeats.Resize(tgt_mat.NumRows(), tgt_mat.NumCols(), kUndefined);
	  tfeats.CopyFromMat(tgt_mat);
	  if (fea_mat.NumRows() < tgt_mat.NumRows())
	    tfeats.Resize(fea_mat.NumRows(), tgt_mat.NumCols(), kCopyData);
	  //else
	  //  tfeats.Resize(tgt_mat.NumRows(), tgt_mat.NumCols(), kCopyData);
	  //KALDI_WARN << " tfeats "<< (tfeats.NumRows()) << " vs. "<< (tfeats.NumCols());
	  feats.Resize(fea_mat.NumRows(), fea_mat.NumCols(), kUndefined);
	  feats.CopyFromMat(fea_mat);
	  if (fea_mat.NumRows() > tgt_mat.NumRows())
	    feats.Resize(tgt_mat.NumRows(), fea_mat.NumCols(), kUndefined);
	  
	  //feats.Resize(tgt_mat.NumRows(), fea_mat.NumCols(), kCopyData);
	  //feats = fea_mat;
	  targets.Resize(tfeats.NumRows(), tfeats.NumCols(), kUndefined);//tgt_mat.NumRows(), tgt_mat.NumCols(), kUndefined);
	  targets.CopyFromMat(tfeats);

	  // Weights
	  Vector<BaseFloat> weights;
	  if (frame_weights != "") {
	    weights = weights_reader.Value();
	  } else { // all per-frame weights are 1.0
	    weights.Resize(feats.NumRows());
	    weights.Set(1.0);
	  }
	  //targets = tgt_mat;
	  // B.P.: possibly apply feature transform on acoustic features (i.e. targets)
	  nnet_transf.Feedforward(targets, &feats_transf);
	  // pass data to randomizers
	  KALDI_ASSERT(feats_transf.NumRows() == targets.NumRows());
	  feature_randomizer.AddData(feats);
	  targets_randomizer.AddData(feats_transf);
	  weights_randomizer.AddData(weights);
	  //weights_randomizer.AddData(weights);
	  num_done++;
        // end when randomizer full
	  //if (feature_randomizer.IsFull()) break;
	  //cache.AddData(feats, feats_transf);
	  //num_done++;
	  /*}*/
	          //report the speed
	  if (num_done % 100 == 0) {
	    time_now = time.Elapsed();
	    KALDI_VLOG(1) << "After " << num_done << " utterances: time elapsed = "
                        << time_now/60 << " min; processed " << total_frames/time_now
                        << " frames per second.";
	  }
	}

        Timer t_features;
        feature_reader.Next(); 
	//targets_reader.Next();
        time_next += t_features.Elapsed();

      }
      // randomize
      if (!crossvalidate && randomize) {
	const std::vector<int32>& mask = randomizer_mask.Generate(feature_randomizer.NumFrames());
        feature_randomizer.Randomize(mask);
        targets_randomizer.Randomize(mask);
	weights_randomizer.Randomize(mask);
	//cache.Randomize();
      }
      for ( ; !feature_randomizer.Done(); feature_randomizer.Next(),
	                                  targets_randomizer.Next(),
	                                  weights_randomizer.Next()) {
	const CuMatrixBase<BaseFloat>& nnet_in = feature_randomizer.Value();
        const CuMatrixBase<BaseFloat>& nnet_tgt = targets_randomizer.Value();
	const Vector<BaseFloat>& frm_weights = weights_randomizer.Value();
	nnet.Propagate(nnet_in, &nnet_out);
        mse.Eval(frm_weights, nnet_out, nnet_tgt, &obj_diff);
        if (!crossvalidate) {
          nnet.Backpropagate(obj_diff, NULL);
        }

	// 1st minibatch : show what happens in network 
        if (kaldi::g_kaldi_verbose_level >= 1 && total_frames == 0) { // vlog-1
          KALDI_VLOG(1) << "### After " << total_frames << " frames,";
          KALDI_VLOG(1) << nnet.InfoPropagate();
          if (!crossvalidate) {
            KALDI_VLOG(1) << nnet.InfoBackPropagate();
            KALDI_VLOG(1) << nnet.InfoGradient();
          }
        }

	// monitor the NN training
        if (kaldi::g_kaldi_verbose_level >= 2) { // vlog-2
          if ((total_frames/25000) != ((total_frames+nnet_in.NumRows())/25000)) { // print every 25k frames
            KALDI_VLOG(2) << "### After " << total_frames << " frames,";
            KALDI_VLOG(2) << nnet.InfoPropagate();
            if (!crossvalidate) {
              KALDI_VLOG(2) << nnet.InfoGradient();
            }
          }
        }
	
        total_frames += nnet_in.NumRows();
      }

      // stop training when no more data
      if (feature_reader.Done()) break;
    }
        // after last minibatch : show what happens in network 
    if (kaldi::g_kaldi_verbose_level >= 1) { // vlog-1
      KALDI_VLOG(1) << "### After " << total_frames << " frames,";
      KALDI_VLOG(1) << nnet.InfoPropagate();
      if (!crossvalidate) {
        KALDI_VLOG(1) << nnet.InfoBackPropagate();
        KALDI_VLOG(1) << nnet.InfoGradient();
      }
    }

    if (!crossvalidate) {
      nnet.Write(target_model_filename, binary);
    }
    
    KALDI_LOG << (crossvalidate?"CROSSVALIDATE":"TRAINING") << " FINISHED " 
              << time.Elapsed()/60 << "min, fps" << total_frames/time.Elapsed()
              << ", feature wait " << time_next << "s"; 

    KALDI_LOG << "Done " << num_done << " files, " << num_no_tgt_mat
              << " with no tgt_mats, " << num_other_error
              << " with other errors.";

    KALDI_LOG << mse.Report();

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif


    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
