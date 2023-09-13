
##### count how many lines
# zcat protein.links.detailed.v11.5.txt.gz | wc -l

# #### view some lines of the gz file
# gzip -cd  protein.links.v11.5.txt.gz | head


gzip -cd protein.links.detailed.v11.5.txt.gz | awk -F' ' '{if($10>950)print}' >> protein.links.detailed.v11.5_filtered.txt

gzip -cd protein.links.v11.5.txt.gz | awk -F' ' '{if($3>950)print}' >> protein.links.v11.5_filtered.txt

# remove useless mutation lines (# start from 1)
sed -i '2,11d' filename

