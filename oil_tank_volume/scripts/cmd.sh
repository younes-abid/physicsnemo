##########################
##### set up obsutil #####
##########################
# 1 check your system architecture and download the correct obsutil
uname -m
# x86_64 = AMD/Intel 64-bit
# i386 or i686 = 32-bit
# aarch64 = ARM 64-bit
# armv7l = ARM 32-bit
# https://support.huaweicloud.com/intl/en-us/utiltg-obs/obs_11_0001.html
wget -P /home/younes.abid/obsutil https://obs-community-intl.obs.ap-southeast-1.myhuaweicloud.com/obsutil/current/obsutil_linux_amd64.tar.gz
tar -xvzf /home/younes.abid/obsutil/obsutil_linux_amd64.tar.gz -C /home/younes.abid/obsutil

# 2 connect to the server and dowload keys
https://console.vb.g42cloud.com/obs/?region=ae-ad-1#/obs/manage/sardata-bayanat/object/list
Account name: g42_it
user name: younes_abid
pass: **************
# top lerightft:younes_abid-->My Credentials-->top left: Access Keys-->Create Access Key
# Download the key file and save it to /home/younes.abid/obsutil/keys 
cd /home/younes.abid/obsutil/obsutil_linux_amd64_5.7.3
./obsutil config -i {your_AK} -k {your_SK} -e obs.ae-ad-1.vb.g42cloud.com
# This action creates a hidden configuration file in your home directory. To take a peek at its contents, use:
cat ~/.obsutilconfig
# check the access
./obsutil ls obs://sardata-bayanat/ --limit=20 -e=obs.ae-ad-1.vb.g42cloud.com

##########################
### Download from obs ####
##########################
# Download data from obs web interface
https://console.vb.g42cloud.com/obs/?region=ae-ad-1#/obs/manage/sardata-bayanat/object/list
Account name: g42_it
user name: younes_abid
pass: **************
obs://sardata-bayanat/ICEYE/labels/oiltank/raw_data

# Copy data
scp -r /Users/younes.abid/Desktop/oil_tank_volume younes.abid@10.120.125.144:'/home/younes.abid/git/physicsnemo/data'

#Download data via cli
./obsutil cp -r -f -u obs://sardata-bayanat/ICEYE/labels/oiltank/ /home/younes.abid/git/physicsnemo/data/oil_tank_volume -e=obs.ae-ad-1.vb.g42cloud.com

##########################
## scp data and output ###
##########################