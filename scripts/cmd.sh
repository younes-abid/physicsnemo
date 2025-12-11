# disk usage
df -h /mnt/storage/younes.abid
du -ahx / --exclude=/tmp | sort -rh -T /dev/shm | head -n 20
du -ahx / --exclude=/tmp | sort -rh -T /System/Volumes | head -n 20


# RAM usage
free -h
htop
apt-get update
apt-get install -y htop glances 

# count files
ls -1 /mnt/storage/younes.abid/physicsnemo/data/custom_data_2/unziped/WRF_2km | wc  -l
ls -1 /mnt/storage/younes.abid/physicsnemo/data/custom_data_2/unziped/wrf_with_rainrate_fog | wc  -l
ls -1 /mnt/storage/younes.abid/physicsnemo/data/custom_data_2/unziped/wrf_with_rainrate_fog | wc  -l
ls -1 /mnt/storage/younes.abid/physicsnemo/data/custom_data_2/ERA5_interpolated_447| wc  -l
ls -1 /mnt/storage/younes.abid/physicsnemo/data/custom_data_2/ERA5_WRF_combined_447| wc  -l
ls -1 /mnt/storage/younes.abid/physicsnemo/data/custom_data_2/ERA5_WRF_combined_concatenated| wc  -l

# rsync data 
rsync -avz --progress --ignore-existing /Users/younes.abid/Desktop/custom_data_2/WRF_2km/ younes.abid@10.120.125.144:/home/younes.abid/git/physicsnemo/data/custom_data_2/unziped
rsync -avz --progress --ignore-existing /Users/younes.abid/Desktop/weather_data/custom_data_2/wrf_with_rainrate_fog younes.abid@10.120.125.144:'/home/younes.abid/git/physicsnemo/data/custom_data_2/unziped'


# unzip the data
unzip '/home/younes.abid/git/physicsnemo/data/custom_data_2/ziped/invariants_2km.nc.zip' -d '/home/younes.abid/git/physicsnemo/data/custom_data_2/unziped'
unzip '/home/younes.abid/git/physicsnemo/data/custom_data_2/ziped/ERA5_Cropped.zip' -d '/home/younes.abid/git/physicsnemo/data/custom_data_2/unziped'

# delete temp files not used in the last 24 hours
find /app/host/tmp -type f -atime +1 -delete

# set symbolic links 
mkdir -p /mnt/storage/younes.abid/physicsnemo/data/custom_data_2
ln -s /mnt/storage/younes.abid/physicsnemo/data/custom_data_2 /home/younes.abid/git/physicsnemo/data/custom_data_2