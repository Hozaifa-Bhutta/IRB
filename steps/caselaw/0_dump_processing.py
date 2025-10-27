"""
Script to process the full-opinions.csv.bz2 file.
The output will be a folder with JSON files, one for each CAP opinion.
Each file will be named "{opinion_id}.json"
"""
import json
import bz2  # Changed from gzip
import csv  # Changed from json lines
import os
import hydra
import sys
import time
from datetime import datetime
from omegaconf import DictConfig
from argparse import ArgumentParser
from tqdm import tqdm
from typing import Optional

# --- Set CSV field size limit ---
# This is critical for reading the 'xml_harvard' field
try:
    max_int = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_int)
            break
        except OverflowError:
            max_int = int(max_int / 10)
    print(f"Set CSV field size limit to: {max_int}")
except Exception as e:
    print(f"Warning: Could not set CSV field size limit. {e}")
# --------------------------------


def write_to_json(data: dict, filepath: str):
    """Simple replacement for the custom utils.generic.write_to_json"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)


def process_opinions_and_write(input_file: str, output_folder: str, max_pages: int, offset: int = 0, start_from: Optional[str] = None) -> None:
    """
    Reads a bzipped CourtListener opinions CSV file and writes each CAP case
    to a separate JSON file in the specified output folder.

    Parameters
    ----------
        input_file : str
            Path to the bzipped opinions CSV file (ends in .csv.bz2).
        output_folder : str
            Path to the folder where the output JSON files will be saved.
        max_pages : int
            Maximum number of opinion pages to process from the dump file.
        offset : int, optional
            Number of pages to skip from the start of the dump file. Defaults to 0.
        start_from : str, optional
            Timestamp string (e.g., "%Y-%m-%d"). Not currently used for this CSV.
    """
    
    assert input_file.endswith(".csv.bz2"), "Input file must be a .csv.bz2 file"

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    length_data = 0 # number of pages written
    count = 0 # number of pages iterated (including those not written due to offset or date filter)

    print("Starting file processing...")
    
    # Open the bz2-compressed file for reading in text mode ("rt")
    with bz2.open(input_file, 'rt', encoding='utf-8') as f:
        
        # Use the CSV module, specifying the quote char and the escape char
        csv_reader = csv.reader(f, quotechar='"', escapechar='\\')
        
        try:
            # Read the header row
            header = next(csv_reader)
            
            # Find the indices of the columns we need
            id_index = header.index("id")
            xml_harvard_index = header.index("xml_harvard")
            date_created_index = header.index("date_created")
            date_modified_index = header.index("date_modified")

        except (StopIteration, ValueError) as e:
            print(f"Error reading header: {e}")
            return

        pbar = tqdm(total=max_pages, desc="Processing opinions")
        
        for idx, row in enumerate(csv_reader):
            if (idx + 1) % 100000 == 0:
                print(f"  ...{idx + 1} rows iterated")

            try:
                # only start writing after 'offset' pages
                if count >= offset:
                    
                    # Get the raw text
                    source = row[xml_harvard_index]
                    
                    # --- This is the main filter ---
                    # If 'xml_harvard' is empty, skip this row
                    if not source:
                        count += 1
                        continue
                    
                    # Extract the rest of the data
                    opinion_id = row[id_index]
                    create_timestamp = row[date_created_index]
                    timestamp = row[date_modified_index]

                    to_write = {
                        "title": opinion_id, # wiki page title (using ID)
                        "source": source, # raw xml text of the opinion
                        "create_timestamp": create_timestamp.replace(" ", "T"), # creation timestamp in CourtListener db
                        "timestamp": timestamp.replace(" ", "T")
                    }
                    
                    try:
                        # write each page to a separate json file, named by its ID
                        filename = f"{opinion_id}.json"
                        output_path = os.path.join(output_folder, filename)
                        write_to_json(to_write, output_path)
                    except FileNotFoundError as e:
                        print(f"Skipping file with invalid name: {opinion_id}. Error: {e}")
                        continue
                    except IOError as e:
                        print(f"Error writing file {opinion_id}.json: {e}")
                        continue
                    
                    length_data += 1
                    pbar.update(1)
            
                count += 1
            except IndexError:
                # This can happen if a row is malformed and shorter than the header
                print(f"Warning: Skipping malformed row {idx + 1}")
                continue
            except Exception as e:
                print(f"Error processing row {idx + 1}: {e}")
                continue

            # stop if we have written 'max_pages' pages
            if length_data == max_pages:
                print(f"Reached max_pages limit of {max_pages}.")
                break
    
    pbar.close()
    print(f"Processing complete. Wrote {length_data} files.")


@hydra.main(version_base=None, config_path="../../conf/steps", config_name=os.getenv("CONFIG_NAME"))
def main(cfg: DictConfig) -> None:

    input_file = cfg.step0.input_file
    output_folder = cfg.step0.output_folder
    offset = cfg.step0.offset
    max_pages = cfg.step0.max_pages
    # 'start_from' is part of the original config but not used in this version
    # start_from = cfg.general.start_from 
    
    print(f"Input file: {input_file}")
    print(f"Output folder: {output_folder}")
    print(f"Max pages: {max_pages}")
    print(f"Offset: {offset}")
    
    assert os.path.exists(input_file), f"Input file not found: {input_file}"

    process_opinions_and_write(
        input_file = input_file,
        output_folder=output_folder,
        max_pages=max_pages,
        offset=offset
        # start_from=start_from # Not used
    )


if __name__ == "__main__":
    main()
