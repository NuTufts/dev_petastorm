import os,sys,argparse

import ROOT as rt
from larlite import larlite
from ublarcvapp import ublarcvapp

rt.gStyle.SetOptStat(0)
rt.gROOT.ProcessLine( "gErrorIgnoreLevel = 3002;" )

import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, StringType

from petastorm.codecs import ScalarCodec, CompressedImageCodec, NdarrayCodec
from petastorm.etl.dataset_metadata import materialize_dataset
from petastorm.unischema import dict_to_spark_row, Unischema, UnischemaField


FlashMatchSchema = Unischema("FlashMatchSchema",[
    UnischemaField('sourcefile', np.string_, (), ScalarCodec(StringType()),  True),
    UnischemaField('run',        np.int32,   (), ScalarCodec(IntegerType()), False),
    UnischemaField('subrun',     np.int32,   (), ScalarCodec(IntegerType()), False),
    UnischemaField('event',      np.int32,   (), ScalarCodec(IntegerType()), False),
    UnischemaField('trackindex', np.int32,   (), ScalarCodec(IntegerType()), False),    
    UnischemaField('array_flashpe', np.float64, (32,), NdarrayCodec(), False),
])


"""
test script that demos the Flash Matcher class.
"""

### DEV INPUTS
#dlmerged = "dlmerged_mcc9_v13_bnbnue_corsika.root"
dlmerged = "dlmerged_mcc9_v13_bnbnue_corsika_v2.root"
triplets = "larmatch_triplettruth_mcc9_v13_bnbnue_corsika_run0001_subrun0001_test.root"

# what we will extract:
# we need to flash match mctracks to optical reco flashes
# interestingly we also have the waveforms here I think
# the triplet truth-matching code should cluster triplets by ancestor ID.
# then we need to voxelize
# then we can store for each entry
#   1) coordinate tensor (N,3)
#   2) charge vector for each filled voxel (N,3)
#   3) flashmatched optical reco vector

# OUTPUT

output_url="file:///tmp/test_flash_dataset"

if os.path.exists(output_url):
    print("output url exists: ",output_url)
    print("remove it.")
    os.system("rm -r %s"%(output_url))


# flashmatch class from ublarcvapp
fmutil = ublarcvapp.mctools.FlashMatcher()
fmutil.setVerboseLevel(1)


# larlite: data interface for:
#  1) mcreco track
#  2) mcreco showers
#  3) opreco
ioll = larlite.storage_manager( larlite.storage_manager.kREAD )
ioll.add_in_filename(  dlmerged )
ioll.open()

#opio = larlite.storage_manager( larlite.storage_manager.kREAD )
#opio.add_in_filename(  args.input_opreco )
#opio.open()

#f = TFile(args.input_voxelfile,"READ")
#print("passed tfile part")

#voxio = larlite.storage_manager( larlite.storage_manager.kREAD )
#voxio.add_in_filename(  args.input_voxelfile )
#voxio.open()

#outio = larlite.storage_manager( larlite.storage_manager.kWRITE )
#outio.set_out_filename(  args.output_file )
#outio.open()

nentries = ioll.get_entries()
print("Number of entries: ",nentries)

#print("Start loop.")

print("isCosmic from ctor: ",fmutil.isCosmic)

#fmutil.initialize( voxio )

# create spark interface
rowgroup_size_mb = 256
spark = SparkSession.builder.config('spark.driver.memory', '2g').master('local[2]').getOrCreate()
sc = spark.sparkContext
# we are going to store 

start_entry = 0
end_entry = 5
if "_v2" in dlmerged:
    start_entry = 10
    end_entry = 15

for ientry in range( start_entry, end_entry ):

    print()
    print("==========================")
    print("===[ EVENT ",ientry," ]===")
    
    ioll.go_to(ientry)
    #opio.go_to(ientry)

    numTracks = fmutil.numTracks( ioll )
    print("numTracks: ",numTracks)
    
    print("start truth-matching process")
    fmutil.process( ioll )

    if True:
        # for debug
        fmutil.printMatches()

    run     = ioll.run_id()
    subrun  = ioll.subrun_id()
    eventid = ioll.event_id()


    # reco flash vectors
    producer_v = ["simpleFlashBeam","simpleFlashCosmic"]
    flash_np_v = {}
    for iproducer,producer in enumerate(producer_v):
        
        flash_beam_v = ioll.get_data( larlite.data.kOpFlash, producer )
    
        for iflash in range( flash_beam_v.size() ):
            flash = flash_beam_v.at(iflash)

            # we need to make the flash vector, the target output
            flash_np = np.zeros( flash.nOpDets() )

            for iopdet in range( flash.nOpDets() ):
                flash_np[iopdet] = flash.PE(iopdet)

            # uboone has 4 pmt groupings
            score_group = {}
            for igroup in range(4):
                score_group[igroup] = flash_np[ 100*igroup: 100*igroup+32 ].sum()
            print(" [",producer,"] iflash[",iflash,"]: ",score_group)
            
            if producer=="simpleFlashBeam":
                flash_np_v[(iproducer,iflash)] = flash_np[0:32]
            elif producer=="simpleFlashCosmic":
                flash_np_v[(iproducer,iflash)] = flash_np[200:232]
                
    with materialize_dataset(spark, output_url, FlashMatchSchema, rowgroup_size_mb):
        
        print("assemble row data")
        iindex = 0
        rows_dd = []
        for k,v in flash_np_v.items():
            print("store flash: ",k," ",v.sum())
            row = {"sourcefile":dlmerged,
                   "run":int(ioll.run_id()),
                   "subrun":int(ioll.subrun_id()),
                   "event":int(ioll.event_id()),
                   "trackindex":int(iindex),
                   "array_flashpe":v}
            rows_dd.append( dict_to_spark_row(FlashMatchSchema,row) )
            iindex += 1
        print("store rows to parquet file")
        spark.createDataFrame(rows_dd, FlashMatchSchema.as_spark_schema() ) \
            .coalesce( 1 ) \
            .write \
            .partitionBy('sourcefile') \
            .mode('append') \
            .parquet( output_url )
        print("spark write operation")


print("=== FIN ==")
