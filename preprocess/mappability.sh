# method1

# download gem from sourceforge https://sourceforge.net/projects/gemlibrary/files/gem-library/Binary%20pre-release%203/GEM-binaries-Linux-x86_64-core_i3-20130406-045632.tbz2/download
# reference https://evodify.com/gem-mappability/
# can not run on apple silicon
wget https://sourceforge.net/projects/gemlibrary/files/gem-library/Binary%20pre-release%203/GEM-binaries-Linux-x86_64-core_i3-20130406-045632.tbz2
tar -xjf GEM.tbz2
vi ~/.bashrc
export PATH=$PATH:/lustre/home/acct-medzy/medzy-cai/software/GEM/bin
source ~/.bashrc



gem-indexer -T 10 -i /lustre/home/acct-medzy/medzy-cai/project/project_tf_dl/tfnet/data/genome/genome.fa -o /lustre/home/acct-medzy/medzy-cai/project/project_tf_dl/tfnet/data/genome/genome.gem_index
# 35 based on the article ”Predicting transcription factor binding in single cells through deep learning“
gem-mappability -T 10 -I genome.gem_index.gem -l 35 -o genome_mappability_35
# Convert GEM mappability to BED
gem-2-wig -I genome.gem_index.gem -i genome_mappability_35.mappability -o genome_mappability_35
# get wigToBigWig from 
# wget http://hgdownload.cse.ucsc.edu/admin/exe/linux.x86_64/wigToBigWig

# get bigWigToBedGraph from 
# wget http://hgdownload.cse.ucsc.edu/admin/exe/linux.x86_64/bigWigToBedGraph

# get bedGraphTobed from 
# wget https://github.com/xuefzhao/Reference.Mappability/raw/master/Scripts/bedGraphTobed

wigToBigWig genome_mappability_35.wig genome_mappability_35.sizes genome_mappability_35.bw
bigWigToBedGraph genome_mappability_35.bw  genome_mappability_35.bedGraph
bedGraphTobed genome_mappability_35.bedGraph genome_mappability_35.bed 0.3

# Merge overlapping intervals in BED
# get combine_overlapping_BEDintervals.py from
# https://github.com/evodify/genotype-files-manipulations/blob/master/combine_overlapping_BEDintervals.py
python ~/git/genotype-files-manipulations/combine_overlapping_BEDintervals.py -i canFam3_mappability_150.bed -o canFam3_mappability_150.merged.bed -v 0
# where -v defines the overhang size between intervals.


# add all other scripts to /lustre/home/acct-medzy/medzy-cai/software/GEM/bin


#----------------------------------------------#
# method2

# GenMap: Ultra-fast Computation of Genome Mappability
# https://github.com/cpockrandt/genmap
./genmap index -F ~/project/project_tf_dl/tfnet/data/genome/genome.fa -I ~/project/project_tf_dl/tfnet/data/genome/genmap_index/ -S 20 -T 10
./genmap map -K 35 -E 2 -I ~/project/project_tf_dl/tfnet/data/genome/genmap_index/ -O ~/project/project_tf_dl/tfnet/data/genome/genmap_output/ -t -w -bg -T 10
wigToBigWig genome.genmap.wig genome.genmap.chrom.sizes genome.genmap.bw





# reverse strand, do all the same manipulate to reverse strand
seqkit seq -r -t genome.fa > complement_genome.fa