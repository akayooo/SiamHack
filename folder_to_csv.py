import os
import argparse
import pandas as pd

def convert_files_to_csv(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        if os.path.isfile(file_path):
            df = pd.read_csv(file_path, sep='\t', header=None, names=['time', 'pressure'])
            output_file = os.path.join(output_folder, filename + '.csv')
            df.to_csv(output_file, index=False)
            print(f"Converted {filename} to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Конвертирует файлы из формата TSV в CSV.")
    parser.add_argument("--input_folder", required=True, help="Путь к папке с исходными файлами.")
    parser.add_argument("--new_folder_name", required=True, help="Имя (или путь) новой папки для сохранения CSV файлов.")
    
    args = parser.parse_args()
    convert_files_to_csv(args.input_folder, args.new_folder_name)

if __name__ == "__main__":
    main()
