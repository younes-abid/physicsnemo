
#FROM ORYX to local machine
rsync -avz --progress --ignore-existing -e "ssh -p 22" younes.abid@10.120.122.41:/spool/PM/my_packages/dsm2radar/data/run2 ~/Desktop/10.120.125.144/data/sar2height #non rescaled data

rsync -avz --progress --ignore-existing -e "ssh -p 22" younes.abid@10.120.122.41:/spool/PM/my_packages/dsm2radar/data/dataset_preparation/preprocessed_images ~/Desktop/10.120.125.144/data/sar2height # rescaled data
mv ~/Desktop/10.120.125.144/data/sar2height/preprocessed_images  ~/Desktop/10.120.125.144/data/sar2height/rescaled_images
mv ~/Desktop/10.120.125.144/data/sar2height/rescaled_images/labels/* ~/Desktop/10.120.125.144/data/sar2height/rescaled_images/ && rmdir ~/Desktop/10.120.125.144/data/sar2height/rescaled_images/labels
mv ~/Desktop/10.120.125.144/data/sar2height/rescaled_images/images/* ~/Desktop/10.120.125.144/data/sar2height/rescaled_images/ && rmdir ~/Desktop/10.120.125.144/data/sar2height/rescaled_images/images


#FROM local machine to 10.120.125.144
rsync -avz --progress --ignore-existing ~/Desktop/10.120.125.144/data/sar2height/run2 younes.abid@10.120.125.144:'/home/younes.abid/git/physicsnemo/data/sar2height' #non rescaled data
rsync -avz --progress --ignore-existing ~/Desktop/10.120.125.144/data/sar2height/preprocessed_images younes.abid@10.120.125.144:'/home/younes.abid/git/physicsnemo/data/sar2height' # rescaled data


#FROM 10.120.125.144 to local machine
rsync -avz --progress --ignore-existing younes.abid@10.120.125.144:'/home/younes.abid/git/physicsnemo/data/sar2height/processed/filtered_patches' ~/Desktop/10.120.125.144/data/sar2height/processed #filtered patches  

# FROM local machine to ORYX
rsync -avz --progress --ignore-existing -e "ssh -p 22" ~/Desktop/10.120.125.144/data/sar2height/processed/filtered_patches younes.abid@10.120.122.41:'/spool/PM/my_packages/dsm2radar/data' #non rescaled data
rsync -avz --progress --ignore-existing ~/Desktop/10.120.125.144/data/sar2height/processed/filtered_patches younes.abid@10.120.122.41:'/spool/PM/my_packages/dsm2radar/data'
rsync -avz --progress --ignore-existing -e "ssh" ~/Desktop/10.120.125.144/data/sar2height/processed/filtered_patches younes.abid@10.120.122.41:'/spool/PM/my_packages/dsm2radar/data' --rsync-path="sudo rsync"


rsync -avz --progress --ignore-existing ~/Desktop/10.120.125.144/data/sar2height/processed/filtered_patches user@10.120.122.41:'/spool/PM/my_packages/dsm2radar/data'