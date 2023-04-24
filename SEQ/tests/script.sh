FILE_SIZES=(5000 10000 20000 40000 80000 100000 120000 140000 160000 180000 200000)

# Define the file suffixes
FILE_SUFFIXES=("5k.txt" "10k.txt" "20k.txt" "40k.txt" "80k.txt" "100k.txt" "120k.txt" "140k.txt" "160k.txt" "180k.txt" "200k.txt")

# Create the files
for i in ${!FILE_SIZES[@]}; do
    touch ${FILE_SUFFIXES[$i]}
done