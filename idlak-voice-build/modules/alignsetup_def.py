#!/usr/bin/python

# Copyright 2013 CereProc Ltd.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

# Takes corpora information and creates input files for kaldi aligner

import sys, os, xml.sax, glob, re

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
SCRIPT_NAME = os.path.splitext(os.path.split(__file__)[1])[0]
DESCRIPTION = 'Takes corpora information and creates input files for kaldi aligner'

# Add to path
sys.path = sys.path + [SCRIPT_DIR + '/../utils']
sys.path = sys.path + [SCRIPT_DIR]

# import voice build utilities
import build_configuration

# sax handler 
class saxhandler(xml.sax.ContentHandler):
    def __init__(self):
        self.id = ''
        self.data = [[]]
        self.ids = []
        self.lex = {}
        self.oov = {}
        
    def startElement(self, name, attrs):
        if name == "fileid":
            newid = attrs['id']
            if self.id and newid != self.id:
                self.data.append([])
                self.id = newid
                self.ids.append(self.id)
            if not self.id:
                self.id = newid
                self.ids.append(self.id)
        if name == "tk":
            try:
                word = attrs['norm'].upper()
            except:
                print "Failed parsing: %s, attrs:" % name, attrs.getNames(), attrs.getValue('wordid')
                raise
            self.data[-1].append(word)
            if not self.lex.has_key(word):
                self.lex[word] = {}
            if attrs.has_key('lts') and attrs['lts'] == 'true':
                self.oov[word] = 1
            if attrs.has_key('altprons'):
                prons = attrs['altprons'].split(', ')
            else:
                prons = [attrs['pron']]
            for p in prons:
                self.lex[word][p] = 1

def kaldidata(datadir, wavdir, spk, flist, force=False):
    if force or not os.path.isdir(os.path.join(datadir, "train")):
        if not os.path.isdir(os.path.join(datadir, "train")):
            os.mkdir(os.path.join(datadir, "train"))
        # setup lookup for wav files in wavdir and in flist if
        # not empty
        valid_ids = {}
        wavs = glob.glob(wavdir + '/*.wav')
        for w in wavs:
            filename = os.path.splitext(os.path.split(w)[1])[0]

            if len(flist):
                if flist.has_key(filename):
                    valid_ids[filename] = 1
            else:
                valid_ids[filename] = 1

            pat = re.match('([a-z][a-z][a-z]_)?([a-z][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9])', filename)

            if pat:
                # Ensure that both potential filenames, with and without speaker prefix are included.
                valid_ids[str('%s_%s' % (spk, pat.group(2)))] = 1
                valid_ids[pat.group(2)] = 1

        # If there are no flist files present in wavdir, we can't proceed.
        if not wavs:
            raise Exception("No files specified in flist are present in %s" % (wavdir))

        # get the speaker id from the lastr file
        spk = filename[:3]
        # load into XML
        p = xml.sax.make_parser()
        handler = saxhandler()
        p.setContentHandler(handler)
        p.parse(open( os.path.join(datadir, "text_norm.xml"),"r"))
        fp = open(os.path.join(datadir, "train", "text"), 'w') 
        for i in range(len(handler.ids)):
            if valid_ids.has_key(handler.ids[i]):
                # If we are forcing beginning and end silences add <SIL>s
                # fp.write("%s <SIL> %s <SIL>\n" % (spk + '_' + handler.ids[i], ' '.join(handler.data[i])))
                fp.write("%s %s\n" % (spk + '_' + handler.ids[i], ' '.join(handler.data[i])))
        fp.close()
        #write wav list and utt 2 spk mapping (all same speaker)
        fp = open(os.path.join(datadir, "train", "wav.scp"), 'w')
        fputt2spk = open(os.path.join(datadir, "train", "utt2spk"), 'w')
        for uttid in handler.ids:
            if valid_ids.has_key(uttid):
                # BUG! Only works id idlakwav is relative path MA140305
                fp.write("%s %s\n" % (spk + '_' + uttid, os.path.join(datadir, wavdir, spk + '_' + uttid + '.wav')))
                fputt2spk.write("%s %s\n" % (spk + '_' + uttid, spk))
        fp.close()
        fputt2spk.close()
        # lexicon and oov have all words for the corpus
        # whether selected or not by flist
        fpoov = open(os.path.join(datadir, "oov.txt"), 'w')
        fplex = open(os.path.join(datadir, "lexicon.txt"), 'w')
        # add oov word and phone (should never be required!
        fplex.write("<OOV> oov\n")
        # If we are forcing beginning and end silences make lexicon
        # entry for <SIL>
        # fplex.write("<SIL> sil\n")
        # write transcription lexicon and oov lexicon for info
        words = handler.lex.keys()
        words.sort()
        phones = {}
        chars = {}
        for w in words:
            prons = handler.lex[w].keys()
            prons.sort()
            utf8w = w.decode('utf8')
            # get all the characters as a check on normalisation
            for c in utf8w:
                chars[c] = 1
            # get phone set from transcription lexicon
            for p in prons:
                pp = p.split()
                for phone in pp:
                    phones[phone] = 1
                fplex.write("%s %s\n" % (w, p))
            if handler.oov.has_key(w):
                fpoov.write("%s %s\n" % (w, prons[0]))
        fplex.close()
        fpoov.close()
        # write phone set
        # Should throw if phone set is not conformant
        # ie. includes sp or ^a-z@
        fp = open(os.path.join(datadir, "nonsilence_phones.txt"), 'w')
        phones = phones.keys()
        phones.sort()
        fp.write('\n'.join(phones) + '\n')
        fp.close()
        # write character set
        fp = open(os.path.join(datadir, "characters.txt"), 'w')
        chars = chars.keys()
        chars.sort()
        fp.write((' '.join(chars)).encode('utf8') + '\n')
        fp.close()
        # silence models
        fp = open(os.path.join(datadir, "silence_phones.txt"), 'w')
        fp.write("sp\nsil\noov\n")
        fp.close()
        # optional silence models
        fp = open(os.path.join(datadir, "optional_silence.txt"), 'w')
        fp.write("sp\n")
        fp.close()
        # an empty file for the kaldi utils/prepare_lang.sh script
        fp = open(os.path.join(datadir, "extra_questions.txt"), 'w')
        fp.close()

def main():
    # process the options based on the default build configuration
    build_conf, parser = build_configuration.get_config(SCRIPT_NAME, DESCRIPTION, SCRIPT_NAME)
    #print 'SEQ', build_conf.dataseq
    # parse commamd line
    if __name__ == '__main__':
        opts, args = parser.parse_args()
        # and load custom configurations
        if opts.bldconf:
            build_conf.parse(opts.bldconf)
        if opts.spkconf:
            build_conf.parse(opts.spkconf)
        else:
            parser.error("Speaker configuration is required e.g. speaker_conf/bdl.xml")
            
        build_conf.updatefromopts(opts)
    # set up logging, check idlak-scratch, check dependencies and build as required
    logger = build_conf.set_build_environment(SCRIPT_NAME)

    # MODULE SPECIFIC CODE
    # get required input files from idlak-data
    kaldisrcdir = os.path.join(build_conf.kaldidir, 'src')
    accdir = os.path.join(build_conf.idlakdata, build_conf.lang, build_conf.acc)
    spkdir = os.path.join(accdir, build_conf.spk)
    outdir = build_conf.outdir
    # get required directories from dependent modules
    # NONE
    # examine module settings and set as appropriate
    # NO MODULE OPTIONS
    # process data
    #run text through the idlak text processing module
    com = '%s/idlaktxpbin/idlaktxp --pretty --tpdb=%s %s %s\n' % (kaldisrcdir,
                                                                  accdir,
                                                                  os.path.join(spkdir, "text.xml"),
                                                                  os.path.join(outdir, "output", "text_norm.xml"))
    logger.log('Info', 'Running normalisation on input xml text: %s' % (com))
    os.system(com)
    # create kaldi required input files (modified from egs/arctic/s1/run.py
    logger.log('Info', 'Creating kaldi input files and train dir')
    wavdir = os.path.join(build_conf.idlakwav, build_conf.lang, build_conf.acc, build_conf.spk, build_conf.srate)
    # use of relative and absolute paths here appears broken MA140305
    kaldidata(os.path.join(outdir, "output"), wavdir, build_conf.spk, build_conf.flist, True)
    # END OF MODULE SPECIFIC CODE
    
    build_conf.end_processing(SCRIPT_NAME)
    
if __name__ == '__main__':
    main()
    
