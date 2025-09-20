# Create the target directory if it doesn't exist
mkdir -p data/raw

# Define the Zenodo record URL and the list of files
BASE_URL=$1
FILES=(
    "circle.csv"
    "gray.csv"
    "green.csv"
    "red.csv"
    "square.csv"
    "table.csv"
)

# Loop through the list and download each file into data/raw/
for FILE in "${FILES[@]}"; do
    echo "Downloading ${FILE}..."
    wget -O "data/raw/${FILE}" "${BASE_URL}/files/${FILE}"
done

echo "All dataset files downloaded successfully to data/raw/"