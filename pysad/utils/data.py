import csv
import hashlib
import json
import os
import shutil
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Iterator
from typing import Any

import numpy as np

from pysad.utils.array_streamer import ArrayStreamer

# The ODDS files of PyOD's mirror at a fixed commit, so that the data cannot change without a
# change in PySAD.
_MIRROR = (
    "https://github.com/yzhao062/pyod/raw/bd110115c680d29561dd7e05d8f4b7af4f3512f7/notebooks/data/"
)
# The SHA-256 checksum of each file, which a download must match.
_CHECKSUMS = {
    "arrhythmia.mat": "e4d47e499591ae31e5d7f0cd983367092353eac240f48e14c9b36457edb3460d",
    "cardio.mat": "5e568b6a73711012c481fa6cc984ab9c68e9a4fa44d6aeb3b43efa381c3f695b",
    "glass.mat": "2c506e6e1e2ca1e82a905d10a90773151ff812c4bd969c71a12db5b2bb6e7921",
    "ionosphere.mat": "8ac5dfa6f47db4acdfe10615489b7f7dd015d79d5b381ce641e72baca3cfefb0",
    "letter.mat": "f4a229b0e2a6b0c9157ce27fa760bcbac0f948cd690b2f5d51c120e5450f2195",
    "lympho.mat": "031b907ecb50347720b08b38f8139d05635adeca9335a064d89d5569c27a5834",
    "mnist.mat": "9479e221f236684c09f3e11c7ad124725d34ecd301b9450425e8b113368413a6",
    "musk.mat": "50bbf414caceb66cfe6f6b1eb7c1ff7c6eb1e6d87173f7bb597ba55cd3dc5493",
    "optdigits.mat": "30c664cd6f3740a173277a27cd1028d471b06f0835a9902c9712749cec58c1c4",
    "pendigits.mat": "7bc32ac83b3d583443acea8b1342a54aaa59eaedb9996af405322f473f3c22bc",
    "pima.mat": "b6a5be8d548b9293d6958beceb1c77055591714d25d40d957eba87808456468f",
    "satellite.mat": "6feac3112b9c14e1c3e60afc437f2f3d29dc1000119c9f950c628078778d6aa0",
    "satimage-2.mat": "95c37a07cea2cb201f6900d14f521510e10b6b2f0074c06ac46d3157440ec67b",
    "shuttle.mat": "b6e396c92354d5dcf5036901e36dc58e2913f2d1a5e138c35a2bedf9c69db899",
    "vertebral.mat": "84b50b055f118f54b5c976650ccd284ab7c1adee822f8b3eb517c7e2c890d15e",
    "vowels.mat": "51d5397bad1ceeda22b7cd44fd58125c5d603c9c2edb53aa59933330e259ac7b",
    "wbc.mat": "afe6b26953cc196b9a480318c903e0edef22529c2f61bd02082438c6e70630a7",
}


# The Numenta Anomaly Benchmark (NAB) at the first commit under the MIT license. Its earlier
# commits are under the AGPL-3.0. Each univariate series is "nab/<category>/<name>.csv".
_NAB = "https://raw.githubusercontent.com/numenta/NAB/ea702d75cc2258d9d7dd35ca8e5e2539d71f3140/"
_NAB_WINDOWS = "1e1fbc4601321aad8d0f8b3784c8134299379f68f6c1f7777565f8ffd57ab6b1"
_NAB_CHECKSUMS = {
    "artificialNoAnomaly/art_daily_no_noise.csv": "ccb7743ac2de8ca110db4af796d11ba31a23ad448a28e353a935c15f637756bc",
    "artificialNoAnomaly/art_daily_perfect_square_wave.csv": "662ff0a295b81eb0ff8e81116a54a497cd6f6b07c19c7a5ea72cce494913ce62",
    "artificialNoAnomaly/art_daily_small_noise.csv": "c4b304229f69f108eaa024fb2406fcd4510c5c5d95cd0053e2da6989682e4e6d",
    "artificialNoAnomaly/art_flatline.csv": "1cc10761ddd6d55fb5f0d9cfc73610416abbb120ee14bb65eff288509a463397",
    "artificialNoAnomaly/art_noisy.csv": "ad455adeaf06852222500955f2aefe6bd389cc762ef68e7102ee543209c8607f",
    "artificialWithAnomaly/art_daily_flatmiddle.csv": "ea5df125947e9ee2b977ac3d341f3c2a029588862af97cae4a654fc56e838702",
    "artificialWithAnomaly/art_daily_jumpsdown.csv": "4af31802e87dff4b4ba3ceb21714ba9d905f7888e1f53d47cc6dd4f856f9debc",
    "artificialWithAnomaly/art_daily_jumpsup.csv": "f79f92b93ee4f9c10e2dc552505e8880b0c618639ed5e0f9816e044b75c26456",
    "artificialWithAnomaly/art_daily_nojump.csv": "65c85d1834376ed5e622f01721d478609577ea0df0a067c5a5ebe5edea03b077",
    "artificialWithAnomaly/art_increase_spike_density.csv": "0959d8871d5c1a3fc3120fb7421e4ef113fffb434af2889085b52aebdb69360c",
    "artificialWithAnomaly/art_load_balancer_spikes.csv": "99f7130e54c7c3e818df8e3bdc0974cd241914bb14a10a9c3fa7c1374125d224",
    "realAWSCloudwatch/ec2_cpu_utilization_24ae8d.csv": "ab446fbd8b9f37507eb2fdb06315826d8daeef02e241133ce06e0ee571ba53d9",
    "realAWSCloudwatch/ec2_cpu_utilization_53ea38.csv": "8942e498de7b40f1b4a6d802755592c09b8fb73ce8f658763a09cb9ea1ea94ac",
    "realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv": "01613e6f632d067f11a5dfd40a188b0789752b388d9bc77a398bd06333878a76",
    "realAWSCloudwatch/ec2_cpu_utilization_77c1ca.csv": "90ceabd570b449241ee24ff8a116a793979b7671ce4490707311bfea0e0aae1f",
    "realAWSCloudwatch/ec2_cpu_utilization_825cc2.csv": "d768419037c9db269343822957314f57ee21a7d9a4d41df2add0d1ba45ba84de",
    "realAWSCloudwatch/ec2_cpu_utilization_ac20cd.csv": "749a15c2e1a4543c21fee9cbf3338cd8a7ed5f5f8a1308b9b099b06c2c66e66b",
    "realAWSCloudwatch/ec2_cpu_utilization_c6585a.csv": "d936cea74682ed43ac96b778352d7de294c0b0c168f4a7e6817162346cdb28c1",
    "realAWSCloudwatch/ec2_cpu_utilization_fe7f93.csv": "f3433f8171f4dcea86c0c7af9996d0f166f812fa0f4567f1d5cd85d2d2cd69b4",
    "realAWSCloudwatch/ec2_disk_write_bytes_1ef3de.csv": "ce3d6c6a652ed7c31a6dd3727a36e128fcac6fb2381c63267ad0fbc02eccd9df",
    "realAWSCloudwatch/ec2_disk_write_bytes_c0d644.csv": "a0e734a66098bb81833835f0839dadafbc8ae829c51494670ee8849e46eb5b63",
    "realAWSCloudwatch/ec2_network_in_257a54.csv": "39104b08f2e0a673b5137eb7681897fcadf0955fedf565740a6a94edc63a81a4",
    "realAWSCloudwatch/ec2_network_in_5abac7.csv": "c27ef93f582af63bf42e80d804764cbe64ae987620f3cfab5477d6ebb3c6157e",
    "realAWSCloudwatch/elb_request_count_8c0756.csv": "74c26574a01ca9fb89dddb5021e2e13c3a93eb25dc640438a9acb1ceb00f1021",
    "realAWSCloudwatch/grok_asg_anomaly.csv": "86a0abe9d58e376858d2fd4f4a438f1997e24c64a09e5af03accd4e1b372ac47",
    "realAWSCloudwatch/iio_us-east-1_i-a2eb1cd9_NetworkIn.csv": "f115fbe6542159eed5612445bfa620708a28ab6b41349d1432bb15c3a3b8414d",
    "realAWSCloudwatch/rds_cpu_utilization_cc0c53.csv": "d5df979ef85928769a016cd0f935919d8c2a113926e21e1eb3a5001d688760e2",
    "realAWSCloudwatch/rds_cpu_utilization_e47b3b.csv": "6b712b922ab3b3404c5a629d64a570c390c99471ee8b9b0ff7665f553fda92c1",
    "realAdExchange/exchange-2_cpc_results.csv": "d869ab165254f3ee77ee21fd9fab2be1d07c1f80b8133cf01bbaf2d768c2ac64",
    "realAdExchange/exchange-2_cpm_results.csv": "1f01b9438566d2fe1ac446faa202a395ac86db9240a1955bc1f118c8eada2a19",
    "realAdExchange/exchange-3_cpc_results.csv": "e70b24dba107a2589657cb3feb9ca5605b4f3f7de49ff445f32f5991bc0bdbe5",
    "realAdExchange/exchange-3_cpm_results.csv": "74d354a94fb1730e14b469c7c1c6e99c2ad69af53afdb85f65220bd3cf01ce4a",
    "realAdExchange/exchange-4_cpc_results.csv": "d4c13abd6f28594dc446f22106562ebe917fdbe76170f60a500c2a402ecc09e2",
    "realAdExchange/exchange-4_cpm_results.csv": "4e5c955b632145b77b93fba4a5e3bf615d47e00416e17091d64e05ae109c760e",
    "realKnownCause/ambient_temperature_system_failure.csv": "230b68ccca20f59d562afd5d24ad52939c9b784386bed0054018358bf9120581",
    "realKnownCause/cpu_utilization_asg_misconfiguration.csv": "58ba65dc0737cfbac11b51514476d50c438d44011232144bb8d93f392df58f9f",
    "realKnownCause/ec2_request_latency_system_failure.csv": "98378580aa80157e057c61d59d81daddccc6c65a2c0c800e3f01f603b8215c3f",
    "realKnownCause/machine_temperature_system_failure.csv": "92bf5b87fc7f9bba8ca0b7ec63ccaac8cb4a1371a258e8c29a10ae9c018d82a4",
    "realKnownCause/nyc_taxi.csv": "d8fa6f7f0734bf5c8be12c52a94e20a82664c397d9dec4449156bd453d32856d",
    "realKnownCause/rogue_agent_key_hold.csv": "47b110baa8e3636574c1033dda1cce780873647fab58223d159d4fee6f180e86",
    "realKnownCause/rogue_agent_key_updown.csv": "9797cfef477c13306f7a33874ab4d6cf700f8f2c7e5825a2a5bc02004370909a",
    "realTraffic/TravelTime_387.csv": "38e70e9ead8227fd023bff6e93247d5bc01ee008f781b921a156962419b399bb",
    "realTraffic/TravelTime_451.csv": "7c174adfc457b201cd111d60688f7ec37eb7201c8080790dc41eeff477bd7345",
    "realTraffic/occupancy_6005.csv": "cd357d7820d675074270fd976d4af1fc1e7854ecb764783028cbcb18d980c91d",
    "realTraffic/occupancy_t4013.csv": "5663a8122a300360eb51fbbd0f21706da05af1af55262926d6a226bb6d071704",
    "realTraffic/speed_6005.csv": "7976e7596cd1e579696c737576e96a26fe041484472ed7d0a463ce2b094d7aa7",
    "realTraffic/speed_7578.csv": "15ba994384730cb5a2e6818f17ded1430b40d92eb7a08e5b1a48a445c8c350f6",
    "realTraffic/speed_t4013.csv": "fa5532d6f7db36cadc73e657fd4dfef05cb1ec44d4010243b314d3f1bbd6a7b5",
    "realTweets/Twitter_volume_AAPL.csv": "826f5cf404c2890784a7824f7102fd00cb134a4948e12e44ec320d095cbbc217",
    "realTweets/Twitter_volume_AMZN.csv": "234f5e63f19a62720dfb7ccfd6ecc8349b94d93b33ef8aafbdb17edda0b654f3",
    "realTweets/Twitter_volume_CRM.csv": "e7133bf26479623d58fc7367107aea1cd7df6020420dece48b0eee4d26cdb28a",
    "realTweets/Twitter_volume_CVS.csv": "f0bdc89cba3dd2265b47069187c750eeae18ef48f76f86067ee26ea1af9c3114",
    "realTweets/Twitter_volume_FB.csv": "a7096bd646759486bda6796b5600cd33a4590f0842bf274ea7acba001b3ba591",
    "realTweets/Twitter_volume_GOOG.csv": "3a39cc23d1ff6a0f234b55298d0c46c21d2e94ddb81a0dad9e1e46e3aaa9ec90",
    "realTweets/Twitter_volume_IBM.csv": "4309bd56d28ddcbdc168e207da9717395fa2e9f74dba41fa768e0f7da8a116fa",
    "realTweets/Twitter_volume_KO.csv": "f3aa8157043c6569911031535bd1f7a73b428b5a2d95238d79b9d5b88ceebbd2",
    "realTweets/Twitter_volume_PFE.csv": "18d2804d6a5dae63fe711baaba9242dce38a32bfade3dcdee1d8a7c1a866e41b",
    "realTweets/Twitter_volume_UPS.csv": "be50f06bbb9d020f37653c8c5cf19af63f77aa0019d8cf10849d91f4af455af4",
}

# The UCR Time Series Anomaly Archive of 2021. Data downloads the archive from the site of its
# authors, because the archive has no license that lets PySAD copy it. The name of each series is
# "<number>_UCR_Anomaly_<name>_<end of training part>_<first anomaly point>_<last anomaly point>.txt".
_UCR = "https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCR_TimeSeriesAnomalyDatasets2021.zip"
_UCR_ARCHIVE = "ucr/UCR_TimeSeriesAnomalyDatasets2021.zip"
_UCR_CHECKSUM = "ac4b991c701e620ae9cc5ebd57ae45593a36cc9c0b6ed5e3c4b7e466cf4783d4"
_UCR_FOLDER = (
    "AnomalyDatasets_2021/UCR_TimeSeriesAnomalyDatasets2021/FilesAreInHere/UCR_Anomaly_FullData/"
)
_UCR_SERIES = """
001_UCR_Anomaly_DISTORTED1sddb40_35000_52000_52620.txt
002_UCR_Anomaly_DISTORTED2sddb40_35000_56600_56900.txt
003_UCR_Anomaly_DISTORTED3sddb40_35000_46600_46900.txt
004_UCR_Anomaly_DISTORTEDBIDMC1_2500_5400_5600.txt
005_UCR_Anomaly_DISTORTEDCIMIS44AirTemperature1_4000_5391_5392.txt
006_UCR_Anomaly_DISTORTEDCIMIS44AirTemperature2_4000_5703_5727.txt
007_UCR_Anomaly_DISTORTEDCIMIS44AirTemperature3_4000_6520_6544.txt
008_UCR_Anomaly_DISTORTEDCIMIS44AirTemperature4_4000_5549_5597.txt
009_UCR_Anomaly_DISTORTEDCIMIS44AirTemperature5_4000_4852_4900.txt
010_UCR_Anomaly_DISTORTEDCIMIS44AirTemperature6_4000_6006_6054.txt
011_UCR_Anomaly_DISTORTEDECG1_10000_11800_12100.txt
012_UCR_Anomaly_DISTORTEDECG2_15000_16000_16100.txt
013_UCR_Anomaly_DISTORTEDECG3_15000_16000_16100.txt
014_UCR_Anomaly_DISTORTEDECG3_8000_17000_17100.txt
015_UCR_Anomaly_DISTORTEDECG4_5000_16800_17100.txt
016_UCR_Anomaly_DISTORTEDECG4_5000_16900_17100.txt
017_UCR_Anomaly_DISTORTEDECG4_5000_17000_17100.txt
018_UCR_Anomaly_DISTORTEDECG4_8000_17000_17100.txt
019_UCR_Anomaly_DISTORTEDGP711MarkerLFM5z1_5000_6168_6212.txt
020_UCR_Anomaly_DISTORTEDGP711MarkerLFM5z2_5000_7175_7388.txt
021_UCR_Anomaly_DISTORTEDGP711MarkerLFM5z3_5000_5948_5993.txt
022_UCR_Anomaly_DISTORTEDGP711MarkerLFM5z4_4000_6527_6645.txt
023_UCR_Anomaly_DISTORTEDGP711MarkerLFM5z5_5000_8612_8716.txt
024_UCR_Anomaly_DISTORTEDInternalBleeding10_3200_4526_4556.txt
025_UCR_Anomaly_DISTORTEDInternalBleeding14_2800_5607_5634.txt
026_UCR_Anomaly_DISTORTEDInternalBleeding15_1700_5684_5854.txt
027_UCR_Anomaly_DISTORTEDInternalBleeding16_1200_4187_4199.txt
028_UCR_Anomaly_DISTORTEDInternalBleeding17_1600_3198_3309.txt
029_UCR_Anomaly_DISTORTEDInternalBleeding18_2300_4485_4587.txt
030_UCR_Anomaly_DISTORTEDInternalBleeding19_3000_4187_4197.txt
031_UCR_Anomaly_DISTORTEDInternalBleeding20_2700_5759_5919.txt
032_UCR_Anomaly_DISTORTEDInternalBleeding4_1000_4675_5033.txt
033_UCR_Anomaly_DISTORTEDInternalBleeding5_4000_6200_6370.txt
034_UCR_Anomaly_DISTORTEDInternalBleeding6_1500_3474_3629.txt
035_UCR_Anomaly_DISTORTEDInternalBleeding8_2500_5865_5974.txt
036_UCR_Anomaly_DISTORTEDInternalBleeding9_4200_6599_6681.txt
037_UCR_Anomaly_DISTORTEDLab2Cmac011215EPG1_5000_17210_17260.txt
038_UCR_Anomaly_DISTORTEDLab2Cmac011215EPG2_5000_27862_27932.txt
039_UCR_Anomaly_DISTORTEDLab2Cmac011215EPG3_5000_16390_16420.txt
040_UCR_Anomaly_DISTORTEDLab2Cmac011215EPG4_6000_17390_17520.txt
041_UCR_Anomaly_DISTORTEDLab2Cmac011215EPG5_7000_17390_17520.txt
042_UCR_Anomaly_DISTORTEDLab2Cmac011215EPG6_7000_12190_12420.txt
043_UCR_Anomaly_DISTORTEDMesoplodonDensirostris_10000_19280_19440.txt
044_UCR_Anomaly_DISTORTEDPowerDemand1_9000_18485_18821.txt
045_UCR_Anomaly_DISTORTEDPowerDemand2_14000_23357_23717.txt
046_UCR_Anomaly_DISTORTEDPowerDemand3_16000_23405_23477.txt
047_UCR_Anomaly_DISTORTEDPowerDemand4_18000_24005_24077.txt
048_UCR_Anomaly_DISTORTEDTkeepFifthMARS_3500_5988_6085.txt
049_UCR_Anomaly_DISTORTEDTkeepFirstMARS_3500_5365_5380.txt
050_UCR_Anomaly_DISTORTEDTkeepForthMARS_3500_5988_6085.txt
051_UCR_Anomaly_DISTORTEDTkeepSecondMARS_3500_9330_9340.txt
052_UCR_Anomaly_DISTORTEDTkeepThirdMARS_3500_4711_4809.txt
053_UCR_Anomaly_DISTORTEDWalkingAceleration1_1500_2764_2995.txt
054_UCR_Anomaly_DISTORTEDWalkingAceleration5_2700_5920_5979.txt
055_UCR_Anomaly_DISTORTEDapneaecg2_10000_20950_21100.txt
056_UCR_Anomaly_DISTORTEDapneaecg3_5000_11111_11211.txt
057_UCR_Anomaly_DISTORTEDapneaecg4_6000_16000_16100.txt
058_UCR_Anomaly_DISTORTEDapneaecg_10000_12240_12308.txt
059_UCR_Anomaly_DISTORTEDgait1_20000_38500_38800.txt
060_UCR_Anomaly_DISTORTEDgait2_22000_46500_46800.txt
061_UCR_Anomaly_DISTORTEDgait3_24500_59900_60500.txt
062_UCR_Anomaly_DISTORTEDgaitHunt1_18500_33070_33180.txt
063_UCR_Anomaly_DISTORTEDgaitHunt2_18500_31200_31850.txt
064_UCR_Anomaly_DISTORTEDgaitHunt3_23400_38400_39200.txt
065_UCR_Anomaly_DISTORTEDinsectEPG1_3000_7000_7030.txt
066_UCR_Anomaly_DISTORTEDinsectEPG2_3700_8000_8025.txt
067_UCR_Anomaly_DISTORTEDinsectEPG3_5200_7000_7050.txt
068_UCR_Anomaly_DISTORTEDinsectEPG4_1300_6508_6558.txt
069_UCR_Anomaly_DISTORTEDinsectEPG5_3200_8500_8501.txt
070_UCR_Anomaly_DISTORTEDltstdbs30791AI_17555_52600_52800.txt
071_UCR_Anomaly_DISTORTEDltstdbs30791AS_23000_52600_52800.txt
072_UCR_Anomaly_DISTORTEDltstdbs30791ES_20000_52600_52800.txt
073_UCR_Anomaly_DISTORTEDpark3m_60000_72150_72495.txt
074_UCR_Anomaly_DISTORTEDqtdbSel1005V_4000_12400_12800.txt
075_UCR_Anomaly_DISTORTEDqtdbSel100MLII_4000_13400_13800.txt
076_UCR_Anomaly_DISTORTEDresperation10_48000_130700_131880.txt
077_UCR_Anomaly_DISTORTEDresperation11_58000_110800_110801.txt
078_UCR_Anomaly_DISTORTEDresperation1_100000_110260_110412.txt
079_UCR_Anomaly_DISTORTEDresperation2_30000_168250_168250.txt
080_UCR_Anomaly_DISTORTEDresperation2_30000_168250_168251.txt
081_UCR_Anomaly_DISTORTEDresperation3_45000_158250_158251.txt
082_UCR_Anomaly_DISTORTEDresperation4_70000_128430_128431.txt
083_UCR_Anomaly_DISTORTEDresperation9_38000_143411_143511.txt
084_UCR_Anomaly_DISTORTEDs20101mML2_12000_35774_35874.txt
085_UCR_Anomaly_DISTORTEDs20101m_10000_35774_35874.txt
086_UCR_Anomaly_DISTORTEDsddb49_20000_67950_68200.txt
087_UCR_Anomaly_DISTORTEDsel840mECG1_17000_51370_51740.txt
088_UCR_Anomaly_DISTORTEDsel840mECG2_20000_49370_49740.txt
089_UCR_Anomaly_DISTORTEDtiltAPB1_100000_114283_114350.txt
090_UCR_Anomaly_DISTORTEDtiltAPB2_50000_124159_124985.txt
091_UCR_Anomaly_DISTORTEDtiltAPB3_40000_114000_114370.txt
092_UCR_Anomaly_DISTORTEDtiltAPB4_20000_67995_67996.txt
093_UCR_Anomaly_NOISE1sddb40_35000_52000_52620.txt
094_UCR_Anomaly_NOISEBIDMC1_2500_5400_5600.txt
095_UCR_Anomaly_NOISECIMIS44AirTemperature4_4000_5549_5597.txt
096_UCR_Anomaly_NOISEECG4_5000_16900_17100.txt
097_UCR_Anomaly_NOISEGP711MarkerLFM5z3_5000_5948_5993.txt
098_UCR_Anomaly_NOISEInternalBleeding16_1200_4187_4199.txt
099_UCR_Anomaly_NOISEInternalBleeding6_1500_3474_3629.txt
100_UCR_Anomaly_NOISELab2Cmac011215EPG1_5000_17210_17260.txt
101_UCR_Anomaly_NOISELab2Cmac011215EPG4_6000_17390_17520.txt
102_UCR_Anomaly_NOISEMesoplodonDensirostris_10000_19280_19440.txt
103_UCR_Anomaly_NOISETkeepThirdMARS_3500_4711_4809.txt
104_UCR_Anomaly_NOISEapneaecg4_6000_16000_16100.txt
105_UCR_Anomaly_NOISEgait3_24500_59900_60500.txt
106_UCR_Anomaly_NOISEgaitHunt2_18500_31200_31850.txt
107_UCR_Anomaly_NOISEinsectEPG3_5200_7000_7050.txt
108_UCR_Anomaly_NOISEresperation2_30000_168250_168250.txt
109_UCR_Anomaly_1sddb40_35000_52000_52620.txt
110_UCR_Anomaly_2sddb40_35000_56600_56900.txt
111_UCR_Anomaly_3sddb40_35000_46600_46900.txt
112_UCR_Anomaly_BIDMC1_2500_5400_5600.txt
113_UCR_Anomaly_CIMIS44AirTemperature1_4000_5391_5392.txt
114_UCR_Anomaly_CIMIS44AirTemperature2_4000_5703_5727.txt
115_UCR_Anomaly_CIMIS44AirTemperature3_4000_6520_6544.txt
116_UCR_Anomaly_CIMIS44AirTemperature4_4000_5549_5597.txt
117_UCR_Anomaly_CIMIS44AirTemperature5_4000_4852_4900.txt
118_UCR_Anomaly_CIMIS44AirTemperature6_4000_6006_6054.txt
119_UCR_Anomaly_ECG1_10000_11800_12100.txt
120_UCR_Anomaly_ECG2_15000_16000_16100.txt
121_UCR_Anomaly_ECG3_15000_16000_16100.txt
122_UCR_Anomaly_ECG3_8000_17000_17100.txt
123_UCR_Anomaly_ECG4_5000_16800_17100.txt
124_UCR_Anomaly_ECG4_5000_16900_17100.txt
125_UCR_Anomaly_ECG4_5000_17000_17100.txt
126_UCR_Anomaly_ECG4_8000_17000_17100.txt
127_UCR_Anomaly_GP711MarkerLFM5z1_5000_6168_6212.txt
128_UCR_Anomaly_GP711MarkerLFM5z2_5000_7175_7388.txt
129_UCR_Anomaly_GP711MarkerLFM5z3_5000_5948_5993.txt
130_UCR_Anomaly_GP711MarkerLFM5z4_4000_6527_6645.txt
131_UCR_Anomaly_GP711MarkerLFM5z5_5000_8612_8716.txt
132_UCR_Anomaly_InternalBleeding10_3200_4526_4556.txt
133_UCR_Anomaly_InternalBleeding14_2800_5607_5634.txt
134_UCR_Anomaly_InternalBleeding15_1700_5684_5854.txt
135_UCR_Anomaly_InternalBleeding16_1200_4187_4199.txt
136_UCR_Anomaly_InternalBleeding17_1600_3198_3309.txt
137_UCR_Anomaly_InternalBleeding18_2300_4485_4587.txt
138_UCR_Anomaly_InternalBleeding19_3000_4187_4197.txt
139_UCR_Anomaly_InternalBleeding20_2700_5759_5919.txt
140_UCR_Anomaly_InternalBleeding4_1000_4675_5033.txt
141_UCR_Anomaly_InternalBleeding5_4000_6200_6370.txt
142_UCR_Anomaly_InternalBleeding6_1500_3474_3629.txt
143_UCR_Anomaly_InternalBleeding8_2500_5865_5974.txt
144_UCR_Anomaly_InternalBleeding9_4200_6599_6681.txt
145_UCR_Anomaly_Lab2Cmac011215EPG1_5000_17210_17260.txt
146_UCR_Anomaly_Lab2Cmac011215EPG2_5000_27862_27932.txt
147_UCR_Anomaly_Lab2Cmac011215EPG3_5000_16390_16420.txt
148_UCR_Anomaly_Lab2Cmac011215EPG4_6000_17390_17520.txt
149_UCR_Anomaly_Lab2Cmac011215EPG5_7000_17390_17520.txt
150_UCR_Anomaly_Lab2Cmac011215EPG6_7000_12190_12420.txt
151_UCR_Anomaly_MesoplodonDensirostris_10000_19280_19440.txt
152_UCR_Anomaly_PowerDemand1_9000_18485_18821.txt
153_UCR_Anomaly_PowerDemand2_14000_23357_23717.txt
154_UCR_Anomaly_PowerDemand3_16000_23405_23477.txt
155_UCR_Anomaly_PowerDemand4_18000_24005_24077.txt
156_UCR_Anomaly_TkeepFifthMARS_3500_5988_6085.txt
157_UCR_Anomaly_TkeepFirstMARS_3500_5365_5380.txt
158_UCR_Anomaly_TkeepForthMARS_3500_5988_6085.txt
159_UCR_Anomaly_TkeepSecondMARS_3500_9330_9340.txt
160_UCR_Anomaly_TkeepThirdMARS_3500_4711_4809.txt
161_UCR_Anomaly_WalkingAceleration1_1500_2764_2995.txt
162_UCR_Anomaly_WalkingAceleration5_2700_5920_5979.txt
163_UCR_Anomaly_apneaecg2_10000_20950_21100.txt
164_UCR_Anomaly_apneaecg3_5000_11111_11211.txt
165_UCR_Anomaly_apneaecg4_6000_16000_16100.txt
166_UCR_Anomaly_apneaecg_10000_12240_12308.txt
167_UCR_Anomaly_gait1_20000_38500_38800.txt
168_UCR_Anomaly_gait2_22000_46500_46800.txt
169_UCR_Anomaly_gait3_24500_59900_60500.txt
170_UCR_Anomaly_gaitHunt1_18500_33070_33180.txt
171_UCR_Anomaly_gaitHunt2_18500_31200_31850.txt
172_UCR_Anomaly_gaitHunt3_23400_38400_39200.txt
173_UCR_Anomaly_insectEPG1_3000_7000_7030.txt
174_UCR_Anomaly_insectEPG2_3700_8000_8025.txt
175_UCR_Anomaly_insectEPG3_5200_7000_7050.txt
176_UCR_Anomaly_insectEPG4_1300_6508_6558.txt
177_UCR_Anomaly_insectEPG5_3200_8500_8501.txt
178_UCR_Anomaly_ltstdbs30791AI_17555_52600_52800.txt
179_UCR_Anomaly_ltstdbs30791AS_23000_52600_52800.txt
180_UCR_Anomaly_ltstdbs30791ES_20000_52600_52800.txt
181_UCR_Anomaly_park3m_60000_72150_72495.txt
182_UCR_Anomaly_qtdbSel1005V_4000_12400_12800.txt
183_UCR_Anomaly_qtdbSel100MLII_4000_13400_13800.txt
184_UCR_Anomaly_resperation10_48000_130700_131880.txt
185_UCR_Anomaly_resperation11_58000_110800_110801.txt
186_UCR_Anomaly_resperation1_100000_110260_110412.txt
187_UCR_Anomaly_resperation2_30000_168250_168250.txt
188_UCR_Anomaly_resperation2_30000_168250_168251.txt
189_UCR_Anomaly_resperation3_45000_158250_158251.txt
190_UCR_Anomaly_resperation4_70000_128430_128431.txt
191_UCR_Anomaly_resperation9_38000_143411_143511.txt
192_UCR_Anomaly_s20101mML2_12000_35774_35874.txt
193_UCR_Anomaly_s20101m_10000_35774_35874.txt
194_UCR_Anomaly_sddb49_20000_67950_68200.txt
195_UCR_Anomaly_sel840mECG1_17000_51370_51740.txt
196_UCR_Anomaly_sel840mECG2_20000_49370_49740.txt
197_UCR_Anomaly_tiltAPB1_100000_114283_114350.txt
198_UCR_Anomaly_tiltAPB2_50000_124159_124985.txt
199_UCR_Anomaly_tiltAPB3_40000_114000_114370.txt
200_UCR_Anomaly_tiltAPB4_20000_67995_67996.txt
201_UCR_Anomaly_CHARISfive_10000_17001_17016.txt
202_UCR_Anomaly_CHARISfive_10411_10998_11028.txt
203_UCR_Anomaly_CHARISfive_10500_10995_11028.txt
204_UCR_Anomaly_CHARISfive_12412_15000_15070.txt
205_UCR_Anomaly_CHARISfive_9812_28995_29085.txt
206_UCR_Anomaly_CHARISten_25130_29080_29140.txt
207_UCR_Anomaly_CHARISten_3165_26929_26989.txt
208_UCR_Anomaly_CHARISten_5130_27929_27989.txt
209_UCR_Anomaly_Fantasia_19000_26970_27270.txt
210_UCR_Anomaly_Italianpowerdemand_36123_74900_74996.txt
211_UCR_Anomaly_Italianpowerdemand_38113_39240_39336.txt
212_UCR_Anomaly_Italianpowerdemand_8913_29480_29504.txt
213_UCR_Anomaly_STAFFIIIDatabase_33211_126920_127370.txt
214_UCR_Anomaly_STAFFIIIDatabase_34211_125720_126370.txt
215_UCR_Anomaly_STAFFIIIDatabase_36276_106720_107370.txt
216_UCR_Anomaly_STAFFIIIDatabase_37216_160720_161370.txt
217_UCR_Anomaly_STAFFIIIDatabase_38211_150720_151370.txt
218_UCR_Anomaly_STAFFIIIDatabase_41117_210720_211370.txt
219_UCR_Anomaly_STAFFIIIDatabase_41612_64632_64852.txt
220_UCR_Anomaly_STAFFIIIDatabase_43217_250720_251370.txt
221_UCR_Anomaly_STAFFIIIDatabase_45616_163632_164852.txt
222_UCR_Anomaly_mit14046longtermecg_56123_91200_91700.txt
223_UCR_Anomaly_mit14046longtermecg_74123_131200_131700.txt
224_UCR_Anomaly_mit14046longtermecg_76123_191200_191700.txt
225_UCR_Anomaly_mit14046longtermecg_81214_143000_143300.txt
226_UCR_Anomaly_mit14046longtermecg_96123_123000_123300.txt
227_UCR_Anomaly_mit14134longtermecg_11231_29000_29100.txt
228_UCR_Anomaly_mit14134longtermecg_11361_47830_47850.txt
229_UCR_Anomaly_mit14134longtermecg_16363_57960_57970.txt
230_UCR_Anomaly_mit14134longtermecg_19363_19510_19610.txt
231_UCR_Anomaly_mit14134longtermecg_8763_47530_47790.txt
232_UCR_Anomaly_mit14134longtermecg_8763_57530_57790.txt
233_UCR_Anomaly_mit14157longtermecg_18913_24500_24501.txt
234_UCR_Anomaly_mit14157longtermecg_18913_24600_24601.txt
235_UCR_Anomaly_mit14157longtermecg_18913_75450_75451.txt
236_UCR_Anomaly_mit14157longtermecg_19313_46350_46390.txt
237_UCR_Anomaly_mit14157longtermecg_19313_89560_90370.txt
238_UCR_Anomaly_mit14157longtermecg_21311_72600_72780.txt
239_UCR_Anomaly_taichidbS0715Master_190037_593450_593514.txt
240_UCR_Anomaly_taichidbS0715Master_240030_884100_884200.txt
241_UCR_Anomaly_taichidbS0715Master_250000_837400_839100.txt
242_UCR_Anomaly_tilt12744mtable_100000_104630_104890.txt
243_UCR_Anomaly_tilt12744mtable_100000_203355_203400.txt
244_UCR_Anomaly_tilt12754table_100013_104630_104890.txt
245_UCR_Anomaly_tilt12754table_100211_270800_271070.txt
246_UCR_Anomaly_tilt12755mtable_100211_270800_271070.txt
247_UCR_Anomaly_tilt12755mtable_50211_121900_121980.txt
248_UCR_Anomaly_weallwalk_2000_4702_4707.txt
249_UCR_Anomaly_weallwalk_2753_8285_8315.txt
250_UCR_Anomaly_weallwalk_2951_7290_7296.txt
""".split()


# KDD Cup 99 from OpenML dataset 1113, the 10% file of the cup in its original order. The stream
# "kddcup99/http.csv" holds its http connections of logged in users, as in scikit-learn's
# fetch_kddcup99, which cannot download from figshare.
_KDDCUP99 = "https://openml.org/data/v1/download/53996/KDDCup99.arff"
_KDDCUP99_ARCHIVE = "kddcup99/KDDCup99.arff"
_KDDCUP99_CHECKSUM = "29d8e2f3d818eebe2d30f931f910c54e72a92d62ae814b6d95e8d5d1441b425e"

# The Server Machine Dataset (SMD) of OmniAnomaly (MIT license) at a fixed commit. Each stream is the
# test half of a machine, which has the labels. The checksums are of the test file and the labels.
_SMD = "https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly/7fb0e0acf89ea49908896bcc9f9e80fcfff6baf4/ServerMachineDataset/"
_SMD_CHECKSUMS = {
    "machine-1-1.txt": (
        "bdb801d8d52ce6b0c4a20c310edf31b5cee9bc82eb84fb38e7d4a19309b16ddb",
        "cddf33441ae480c6074e173c897d4715eb893f24a7078d3d3e1b79b200c7cd19",
    ),
    "machine-1-2.txt": (
        "f409991af7c049d3f286a8cf89dc2fa99114bea8753f6c3aea4bd24cb03eb015",
        "c2dbb3e4f89c9eb48f30fd1ddae48b80e7ae7ad7ca9abf2ffd71b6557ef2bb20",
    ),
    "machine-1-3.txt": (
        "f3c134d45ffa7aed66587dba893111396c8c7ea3b6fa7f76895955b26b1a75ef",
        "755d57179625682172ae85f9483749cff576ff3282d0af294271a40ab6ea7b28",
    ),
    "machine-1-4.txt": (
        "98ade00e57eac63e8f81a506a8e848803b58590835ab2b3cf53d1f0459900bd3",
        "a470a3f7b66ec3d080fc5ee3d9565e0f3bfcfa16a1650259f771736327fa9a8b",
    ),
    "machine-1-5.txt": (
        "98b8979f21b4b68c7308f38dd093b9bff4275814578365103c692f28f60f61c5",
        "e7e2e611e12b5e9e2558ed955ad271f1f408fbdd7fbbfdfc72eb745ade567815",
    ),
    "machine-1-6.txt": (
        "0797fe4bfd25ef13c830955cb9f2a36b4c9d5da265b8e91702408b6b048267f4",
        "8359ffa67b5421b0b9967ccf396a58c7b3bf52fa24c3f95e6b9df6486f1d6365",
    ),
    "machine-1-7.txt": (
        "785a3f542ef6ac171a3af55c534f9ca598d53fa0371010172c72e33af4588d16",
        "24b3ce00bf2c56b571775a2b28d4e50ed27d050d77a031fa16c11ef3e51809b8",
    ),
    "machine-1-8.txt": (
        "b8784c3b4169f8a0a2ed9f47b00070d72ed214dae0307553b8cfdb80c712d779",
        "8019476032c4c5b4b80127cfb9e74ee7d57fbbd558aa15097518713e16463a47",
    ),
    "machine-2-1.txt": (
        "2d103661799271be958227907950a1c76803356442d60a4070671cc0abbefb20",
        "416dd20a447f38b8d9394382293ed44f509c1c3f99c4aa2fbff179498626dbdc",
    ),
    "machine-2-2.txt": (
        "1cb0dd0a4b8738b184c62bb07700e6800e029f7a76a715fe0ee147c603b2f2c8",
        "b09e71b3aafe71d9ee73a4058b30efcf67dc80df98d3b7b7a8ec019ea3b769c4",
    ),
    "machine-2-3.txt": (
        "eaa24c78ff438eec2f847c09eb4de6ab8dbe8a0bb8d77b52ac4ecdee5bd21a7d",
        "5768ce5280cb5ce4bee6d8bb3c7105819574d5e82f7d517d61495efa9d4f566e",
    ),
    "machine-2-4.txt": (
        "85ba3101b9ab6a7684cd897b2d4c78e85caa6f5ef1c4f0e153479f4fca01cf38",
        "c26a6dabfd2ab4aec34c40919100e13e52e17ad0e6b702dd038f582b2c633b01",
    ),
    "machine-2-5.txt": (
        "842588acfb9299735a785757fcadd98cc7f69b6a511d28e54e1af05c5e95137b",
        "9966e4092bece4e0ef60ef54d85396c49c22da64334aad82a9fc2f3842d2ef15",
    ),
    "machine-2-6.txt": (
        "c895144b7c7fb1ff960f506132d5aeefebce4325efad908d071fb7df5b3b1cf1",
        "6fe3dd209542dc6a20879618c40ef6f5d55ce60f2da2902c311d9f0fff3c1ee0",
    ),
    "machine-2-7.txt": (
        "58c3dfd109437ccbedab9d36608d32874c2ae5473484c5941f85349baf32cef8",
        "d9d7aceca750277dd28883f7b5b34da3653652ea9c96de78baa84daeebc4a9b7",
    ),
    "machine-2-8.txt": (
        "846207955c546f0c3ea57aa7032b8e3c207ad67632a0605c8273f182059d1a72",
        "22fbe94e06e56b6035fcc10383bac90d84ff4604f9063009a3940747f92d8105",
    ),
    "machine-2-9.txt": (
        "8d93ccd1a6606de3f665cb2e905d206b60ad2b2a58f47e9561b9bb67a2cd4f28",
        "eab9425847aa5c7c63485047a49b1bfc9cf44ed88a0d1be84e2fd0a16d0ba895",
    ),
    "machine-3-1.txt": (
        "b110b204ba4919250351eba7aa253d73ab54af0cf18e272b9b284de5fa7df88d",
        "549348a9c63ed922e85157885bcb21289a8c9f5af94b50147a21cb7b928220ec",
    ),
    "machine-3-2.txt": (
        "5f83ded5fbfb9b682030bdd7dd82eb53a77f9caa3773ddcc7e38db73c25db46e",
        "1bbf4223cc6083f718f1d6022e2a48d48af44b08cdfa80048aa51af34d5111c3",
    ),
    "machine-3-3.txt": (
        "ecc3140ce24869a44ebac9b03ca372fb1a6eb7e5bde96946f1a350283d4b152a",
        "4e2c657f613759f11f167c5540d047723d41d8a3c76467fa4daafcc8a7112035",
    ),
    "machine-3-4.txt": (
        "07e1b43e20b7d8300fcd4b4f45ea47ea8c851f3dbd7d1fe971139c32f039357e",
        "5c1f09b85c53d1936eff1e044c5f7413b07f10fd641934b4aba9b2669485b7be",
    ),
    "machine-3-5.txt": (
        "85dbb13b3076981ad917f42081b71a1440d8b2aafd9bee626f0d1cacff43fcd6",
        "8a17a0dff8cc0062124997d665a2055cc1c08daffffb9a63ccfd21f7d10bea8f",
    ),
    "machine-3-6.txt": (
        "04f746a4dde41adbfe5727c6ff77839d1b0b21d6d65bdb0253557e0391c5d44c",
        "2bef9db1d998172c61008f593a7537377179ef4f24963001203c1e48345e9b61",
    ),
    "machine-3-7.txt": (
        "a4588c1f39905d03bd24fce65052e7b175fba336d081843325ef780c9ed8cdad",
        "80b3a19dfd0af2d800c2388474543bdcb723ba582b0f4a986d68f920b757c73c",
    ),
    "machine-3-8.txt": (
        "fa79d5dad3a703fbc13faf0eab26d9466b74789f2bb0d22776f2adaa4236f085",
        "dc834a5b5e9a194f5367423c29410afcb622e0e47d7d3362979dd280de713220",
    ),
    "machine-3-9.txt": (
        "5a8968c269a8a3562e54235aa05344b262f44829c2bae04cf9f5d364f388310d",
        "bdaf308affb4653e3e5826177e7e27c883294d2fa8cf48364850ef5a4dac2282",
    ),
    "machine-3-10.txt": (
        "febaf197d4a8864f3a70a5e5c0926d6dab8f1b515cf0d90d31ee81a22d80392f",
        "d2b87c9c1d1950f0b29eaca0edf4706b917984799f6d9b5c1472435832fe3abd",
    ),
    "machine-3-11.txt": (
        "c62f52e65d1b892ef8c556b38aa5c1d3504cf0cdd97a94e7df08985b922d7af0",
        "ce306777bd66e9ea81de726018e440d49afcc98415cb0db7d99367dddf398f1b",
    ),
}

# The INSECTS streams (CC BY 4.0) as the ARCUS paper published them on Google Drive, with the
# southern house mosquito as the anomaly class. Each entry has the Drive id and the checksum.
_INSECTS = "https://drive.usercontent.google.com/download?export=download&confirm=t&id="
_INSECTS_FILES = {
    "abrupt.csv": (
        "1C9_sXLa-dpA-p5svVYvQB32VANK_0ueD",
        "fbeea8c2456690e0a03378788c25778193df8ea4177393892667b661fb9f32a2",
    ),
    "incremental.csv": (
        "1CBJnaM9jiinpDF3yz_WoIlouurnzoe5v",
        "f4c76ac6049d0e8f780f4a4c8a1aa573fe7049c3df0c6f1874a0e55fdacb67c7",
    ),
    "incremental_gradual.csv": (
        "1CKThZMJkcC7u5T3S8BZukZRu1SXZ_1qS",
        "f9f1ae0e5c9c6c2e8da716f9a77ae17cadd3eb53d7c79ac00646bf2b0b790479",
    ),
    "incremental_reoccurring.csv": (
        "1CHEhJ2-XRk9T1o53PjT49O9mBWopKcPG",
        "fa97083433d50675835f1f78223fba2ffbf1f80729764e8017be1824f436d95e",
    ),
}
_SOURCES = ("odds", "nab", "ucr", "kddcup99", "smd", "insects")

# A download that fails with a server error or a network error is tried again, after 5 and then 10
# seconds.
_ATTEMPTS = 3
_RETRY_SECONDS = 5.0


def _source(name: str) -> tuple[str, str]:
    """The URL and the SHA-256 checksum of a remote file of a dataset."""
    if name == "nab/combined_windows.json":
        return _NAB + "labels/combined_windows.json", _NAB_WINDOWS
    if name.startswith("nab/"):
        return _NAB + "data/" + name[len("nab/") :], _NAB_CHECKSUMS[name[len("nab/") :]]
    if name == _UCR_ARCHIVE:
        return _UCR, _UCR_CHECKSUM
    if name == _KDDCUP99_ARCHIVE:
        return _KDDCUP99, _KDDCUP99_CHECKSUM
    if name.startswith("smd/labels/"):
        return _SMD + "test_label/" + name[len("smd/labels/") :], _SMD_CHECKSUMS[
            name[len("smd/labels/") :]
        ][1]
    if name.startswith("smd/"):
        return _SMD + "test/" + name[len("smd/") :], _SMD_CHECKSUMS[name[len("smd/") :]][0]
    if name.startswith("insects/"):
        drive_id, checksum = _INSECTS_FILES[name[len("insects/") :]]
        return _INSECTS + drive_id, checksum

    return _MIRROR + name, _CHECKSUMS[name]


def _get_default_data_dir() -> str:
    """Returns the default directory for caching datasets.

    Defaults to the directory in the ``PYSAD_DATA_DIR`` environment variable if
    set, or ``~/.cache/pysad``.
    """
    return os.environ.get(
        "PYSAD_DATA_DIR", os.path.expanduser(os.path.join("~", ".cache", "pysad"))
    )


class Data:
    """A helper class to load and stream anomaly detection datasets.

    Datasets are loaded from the local filesystem or downloaded on demand. The multivariate
    datasets come from the Outlier Detection DataSets (ODDS) repository mirror
    (:cite:`rayana2016odds`, hosted in the PyOD repository at a fixed commit). The univariate
    series come from the Numenta Anomaly Benchmark (NAB) and the UCR Time Series Anomaly Archive.
    The multivariate streams in time order come from KDD Cup 99, the Server Machine Dataset (SMD)
    and INSECTS. Each download must match a SHA-256 checksum. :meth:`datasets` lists the datasets, and
    :meth:`get_stream` gives the stream of a run of the PySAD benchmark.

    Downloaded datasets are cached in a local directory. The cache directory
    defaults to ``~/.cache/pysad``, which can be overridden via the
    ``PYSAD_DATA_DIR`` environment variable or by passing an explicit
    ``data_base_path``.

    Args:
        data_base_path (str or None): Base directory that contains the data files and caches the downloaded datasets. If None, the class uses the directory in ``PYSAD_DATA_DIR`` if set, or ``~/.cache/pysad`` (Default=None).
    """

    def __init__(self, data_base_path: str | None = None) -> None:
        self.data_base_path = _get_default_data_dir() if data_base_path is None else data_base_path
        self._is_default_path = data_base_path is None

    @staticmethod
    def datasets(source: str = "odds") -> list[str]:
        """Returns the names of the datasets that this class can download.

        Args:
            source (str): The collection of the datasets (Default="odds"). These are the collections:

                - ``"odds"``: the 17 multivariate datasets of ODDS, for example ``"cardio.mat"``.
                - ``"nab"``: the 58 univariate series of the Numenta Anomaly Benchmark, for example ``"nab/realKnownCause/nyc_taxi.csv"``. The points in an anomaly window of NAB have the label 1.
                - ``"ucr"``: the 250 univariate series of the UCR Time Series Anomaly Archive, for example ``"ucr/025_UCR_Anomaly_DISTORTEDInternalBleeding14_2800_5607_5634.txt"``. The last two numbers of a name are the first and the last point of the anomaly, counted from 1. These points have the label 1. The number before them is the length of the training part, which has no anomalies. The first series download the archive of 184 MB.
                - ``"kddcup99"``: ``"kddcup99/http.csv"``, the 58,724 http connections of logged in users of the KDD Cup 99 10% file, in their original order. The 3 features are ``log(x + 0.1)`` of the duration, the bytes from the source and the bytes to the destination. The attacks have the label 1. The first use downloads the file of 74 MB.
                - ``"smd"``: the test halves of the 28 machines of the Server Machine Dataset, for example ``"smd/machine-1-1.txt"``, with one point of 38 features each minute.
                - ``"insects"``: the 4 INSECTS streams of the ARCUS paper, for example ``"insects/abrupt.csv"``, with 33 features. The southern house mosquito is the anomaly class.

        Returns:
            list[str]: The names of the datasets, which :meth:`get_data` accepts.

        Raises:
            ValueError: If ``source`` is not one of the collections above.
        """
        if source == "odds":
            return list(_CHECKSUMS)
        if source == "nab":
            return ["nab/" + name for name in _NAB_CHECKSUMS]
        if source == "ucr":
            return ["ucr/" + name for name in _UCR_SERIES]
        if source == "kddcup99":
            return ["kddcup99/http.csv"]
        if source == "smd":
            return ["smd/" + name for name in _SMD_CHECKSUMS]
        if source == "insects":
            return ["insects/" + name for name in _INSECTS_FILES]
        raise ValueError(f"source must be one of {_SOURCES}, got {source!r}.")

    def _download(self, file_name: str, target_path: str) -> None:
        """Downloads a dataset file from the remote mirror into target_path atomically.

        Args:
            file_name (str): Name of the remote file (e.g. 'cardio.mat').
            target_path (str): Destination file path.
        """
        url, expected = _source(file_name)
        target_dir = os.path.dirname(target_path)
        if target_dir:
            os.makedirs(target_dir, exist_ok=True)

        temp_file = tempfile.NamedTemporaryFile(
            dir=target_dir or None, prefix=f"tmp_{os.path.basename(file_name)}_", delete=False
        )
        temp_path = temp_file.name
        try:
            temp_file.close()
            req = urllib.request.Request(url, headers={"User-Agent": "pysad"})
            for attempt in range(_ATTEMPTS):
                try:
                    with urllib.request.urlopen(req) as response, open(temp_path, "wb") as f:
                        shutil.copyfileobj(response, f)
                    break
                except urllib.error.URLError as error:
                    client_error = isinstance(error, urllib.error.HTTPError) and error.code < 500
                    if client_error or attempt == _ATTEMPTS - 1:
                        raise
                    time.sleep(_RETRY_SECONDS * 2**attempt)
            with open(temp_path, "rb") as f:
                checksum = hashlib.sha256(f.read()).hexdigest()
            if checksum != expected:
                raise ValueError(
                    f"The download of '{file_name}' has the SHA-256 checksum {checksum}, "
                    f"but PySAD expects {expected}."
                )
            os.replace(temp_path, target_path)
        except Exception:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            raise

    def _load_via_txt(self, path: str) -> np.ndarray:
        """Loads the data file from .txt file.

        Args:
            path (str): The path of data.

        Returns:
            np.float64 array of shape (num_instances, num_features): Feature vectors.
        """
        X = np.loadtxt(path, delimiter=",")

        return X

    def get_data(self, data_file: str, download: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Loads the data given the filename or path, downloading it if not present.

        Args:
            data_file (str): Path or filename of the data.
            download (bool): Whether to download the dataset if not present locally (Default=True).

        Returns:
            tuple[np.ndarray, np.ndarray]: The feature vectors of shape (num_instances, num_features) and the labels of shape (num_instances,).
        """
        if os.path.exists(data_file):
            data_path = data_file
        else:
            data_path = os.path.join(self.data_base_path, data_file)

        # An ODDS dataset is known by its file name, and a stream of another collection by its full name.
        series = data_file.startswith(tuple(f"{source}/" for source in _SOURCES[1:]))
        file_name = data_file if series else os.path.basename(data_file)

        if not os.path.exists(data_path):
            if self._is_default_path:
                local_fallback = os.path.join("data", data_file)
                if os.path.exists(local_fallback):
                    data_path = local_fallback
                else:
                    examples_fallback = os.path.join("examples", "data", data_file)
                    if os.path.exists(examples_fallback):
                        data_path = examples_fallback

        if not os.path.exists(data_path):
            if not download:
                raise FileNotFoundError(
                    f"Dataset file '{data_file}' was not found at '{data_path}' and download=False."
                )

            if not any(file_name in self.datasets(source) for source in _SOURCES):
                raise FileNotFoundError(
                    f"Dataset file '{data_file}' was not found at '{data_path}' and is not a known remote dataset. "
                    f"Available datasets: {self.datasets()}, and the streams of datasets(source) for each source in {_SOURCES[1:]}."
                )

            if file_name.startswith("ucr/"):
                self._extract_ucr(file_name, data_path)
            elif file_name.startswith("kddcup99/"):
                self._extract_kddcup99(data_path)
            else:
                self._download(file_name, data_path)

        if file_name.startswith("nab/"):
            return self._load_nab(file_name, data_path, download)
        if file_name.startswith("ucr/"):
            return self._load_ucr(file_name, data_path)
        if file_name.startswith("smd/"):
            return self._load_smd(file_name, data_path, download)
        if data_path.endswith(".mat"):
            from scipy.io import loadmat

            f = loadmat(data_path)

            X = f["X"]
            y = f["y"].ravel()
        else:
            X = self._load_via_txt(data_path)

            y = X[:, -1].ravel()
            X = X[:, :-1]

        return X, y

    def _load_nab(self, name: str, data_path: str, download: bool) -> tuple[np.ndarray, np.ndarray]:
        """Loads a NAB series and labels the points in its anomaly windows with 1."""
        windows_path = os.path.join(self.data_base_path, "nab", "combined_windows.json")
        if not os.path.exists(windows_path):
            if not download:
                raise FileNotFoundError(
                    f"The NAB labels were not found at '{windows_path}' and download=False."
                )
            self._download("nab/combined_windows.json", windows_path)
        with open(windows_path) as f:
            windows = json.load(f)[name[len("nab/") :]]
        with open(data_path, newline="") as f:
            rows = list(csv.reader(f))[1:]  # the header is "timestamp,value"

        times = np.array([row[0] for row in rows], dtype="datetime64[s]")
        y = np.zeros(len(rows), dtype=int)
        for start, end in windows:
            y[(times >= np.datetime64(start)) & (times <= np.datetime64(end))] = 1

        return np.array([[float(row[1])] for row in rows]), y

    def _extract_ucr(self, name: str, data_path: str) -> None:
        """Copies a UCR series out of the archive, and downloads the archive if it is not present."""
        archive = os.path.join(self.data_base_path, _UCR_ARCHIVE)
        if not os.path.exists(archive):
            self._download(_UCR_ARCHIVE, archive)
        with zipfile.ZipFile(archive) as z, z.open(_UCR_FOLDER + name[len("ucr/") :]) as f:
            content = f.read()
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        with open(data_path, "wb") as f:
            f.write(content)

    def _load_ucr(self, name: str, data_path: str) -> tuple[np.ndarray, np.ndarray]:
        """Loads a UCR series and labels the points of its anomaly with 1."""
        with open(data_path) as f:
            X = np.array(f.read().split(), dtype=np.float64).reshape(
                -1, 1
            )  # some files use one line
        first, last = map(int, name.removesuffix(".txt").split("_")[-2:])
        y = np.zeros(len(X), dtype=int)
        y[first - 1 : last] = 1

        return X, y

    def _extract_kddcup99(self, data_path: str) -> None:
        """Writes the http stream of KDD Cup 99, and downloads the file of the cup if it is not present."""
        archive = os.path.join(self.data_base_path, _KDDCUP99_ARCHIVE)
        if not os.path.exists(archive):
            self._download(_KDDCUP99_ARCHIVE, archive)
        with open(archive) as f:
            lines = (line for line in f if line.strip() and not line.startswith(("@", "%")))
            # The columns 0, 2, 4, 5 and 11 are the duration, the service, the bytes from the
            # source, the bytes to the destination and logged_in. The last one is the label.
            rows = [r for r in csv.reader(lines) if r[2] == "http" and r[11] == "1"]
        X = np.log(np.array([[r[0], r[4], r[5]] for r in rows], dtype=np.float64) + 0.1)
        y = np.array([r[-1] != "normal" for r in rows], dtype=np.float64)
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        np.savetxt(data_path, np.column_stack([X, y]), delimiter=",", fmt="%.17g")

    def _load_smd(self, name: str, data_path: str, download: bool) -> tuple[np.ndarray, np.ndarray]:
        """Loads the test half of an SMD machine and its labels."""
        labels_name = "smd/labels/" + name[len("smd/") :]
        labels_path = os.path.join(self.data_base_path, labels_name)
        if not os.path.exists(labels_path):
            if not download:
                raise FileNotFoundError(
                    f"The SMD labels were not found at '{labels_path}' and download=False."
                )
            self._download(labels_name, labels_path)

        return np.loadtxt(data_path, delimiter=","), np.loadtxt(labels_path, dtype=int)

    def get_iterator(
        self,
        data_file: str,
        shuffle: bool = True,
        random_state: int | np.random.RandomState | None = None,
        download: bool = True,
    ) -> Iterator[np.ndarray | tuple[np.ndarray, Any]]:
        """Returns an iterator over the instances and labels of a dataset, in shuffled order by default.

        Args:
            data_file (str): Path or filename of data.
            shuffle (bool): Whether to shuffle the instances (Default=True).
            random_state (int, np.random.RandomState or None): Seed or random number generator for the shuffle, as in :class:`ArrayStreamer`. None draws from NumPy's global random state (Default=None).
            download (bool): Whether to download the dataset if not present locally (Default=True).

        Returns:
            Iterator[tuple[np.ndarray, Any]]: Iterator over ``(X, y)`` pairs, where ``X`` is the feature vector of shape (num_features,) and ``y`` is its label.
        """
        X, y = self.get_data(data_file, download=download)

        return ArrayStreamer(shuffle=shuffle, random_state=random_state).iter(X, y)

    def describe(self, data_file: str, download: bool = True) -> dict[str, Any]:
        """Returns the number of points, the number of features and the anomaly rate of a dataset.

        Args:
            data_file (str): Path or filename of data.
            download (bool): Whether to download the dataset if not present locally (Default=True).

        Returns:
            dict: ``points`` (int), ``features`` (int) and ``anomaly_rate`` (float, the fraction of points with label 1).
        """
        X, y = self.get_data(data_file, download=download)

        return {"points": len(X), "features": X.shape[1], "anomaly_rate": float(np.mean(y))}

    def get_stream(
        self,
        data_file: str,
        random_state: int | np.random.RandomState | None = None,
        max_points: int | None = None,
        scale: bool = False,
        download: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns a shuffled dataset as arrays, in the order of :meth:`get_iterator`.

        With ``random_state=seed``, ``max_points=5000`` and ``scale=True``, this is the stream of the PySAD benchmark for that seed.

        Args:
            data_file (str): Path or filename of data.
            random_state (int, np.random.RandomState or None): Seed or random number generator for the shuffle. None draws from NumPy's global random state (Default=None).
            max_points (int or None): The number of first points of the shuffled stream to keep. None keeps all of them (Default=None).
            scale (bool): Whether to scale each feature to the range from 0 to 1, with the minimum and the maximum of the points that it keeps. A constant feature becomes 0 (Default=False).
            download (bool): Whether to download the dataset if not present locally (Default=True).

        Returns:
            tuple[np.ndarray, np.ndarray]: The feature vectors of shape (num_points, num_features) and the labels of shape (num_points,), in stream order.
        """
        stream = list(self.get_iterator(data_file, random_state=random_state, download=download))
        stream = stream[:max_points]
        X, y = np.array([x for x, _ in stream]), np.array([label for _, label in stream])
        if scale:
            mins, maxes = X.min(axis=0), X.max(axis=0)
            X = (X - mins) / np.where(maxes > mins, maxes - mins, 1.0)

        return X, y
