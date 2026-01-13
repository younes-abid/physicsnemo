# disk usage
df -h /mnt/storage/younes.abid
du -ahx / --exclude=/tmp | sort -rh -T /dev/shm | head -n 20
du -ahx / --exclude=/tmp | sort -rh -T /System/Volumes | head -n 20

# delete temp files not used in the last 24 hours
find /app/host/tmp -type f -atime +1 -delete