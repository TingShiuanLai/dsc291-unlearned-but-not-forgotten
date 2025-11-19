import json
import os
import uuid

def make_client_names_individually_unique(input_file_path, output_file_path):
    """
    Reads a JSON Lines file and assigns a NEW unique UUID suffix (e.g., -a1b2c3d4) 
    to every occurrence of a client name after the first one, treating them as 
    separate individuals in the synthetic dataset.

    Args:
        input_file_path (str): The path to the original JSON Lines file.
        output_file_path (str): The path where the modified data will be saved.
    """
    # Dictionary to track the count for each name encountered
    # We still need this to know if it's the 1st, 2nd, 3rd time, etc.
    name_count = {}
    
    print(f"Reading from: {input_file_path}")
    print(f"Writing to: {output_file_path}")

    # Ensure the input file exists
    if not os.path.exists(input_file_path):
        print(f"Error: Input file not found at {input_file_path}")
        return

    try:
        with open(input_file_path, 'r', encoding='utf-8') as infile, \
             open(output_file_path, 'w', encoding='utf-8') as outfile:
            
            for line_number, line in enumerate(infile, 1):
                try:
                    record = json.loads(line)
                    client_name = record.get("client_name")
                    
                    if client_name is None:
                        outfile.write(line)
                        continue

                    # --- Logic to assign a new UUID to every occurrence after the first ---
                    
                    # Initialize or increment the count for the current name
                    current_count = name_count.get(client_name, 0)
                    
                    if current_count == 0:
                        # 1st time seeing this name: keep the original name
                        name_count[client_name] = 1
                    else:
                        # 2nd, 3rd, or subsequent time: generate a NEW unique ID
                        
                        # Generate a short 8-character UUID
                        unique_id = str(uuid.uuid4())[:8]
                        
                        # Update the name in the record
                        record["client_name"] = f"{client_name}-{unique_id}"
                        
                        # Increment the counter for the next time this name appears
                        name_count[client_name] += 1
                        
                    # Write the (potentially modified) record to the output file
                    outfile.write(json.dumps(record) + '\n')

                except json.JSONDecodeError:
                    print(f"Warning: Skipping line {line_number} due to invalid JSON.")
                    continue
                
    except Exception as e:
        print(f"An error occurred: {e}")
        return

    print("\nProcessing complete. Each duplicate client name now represents a unique individual with a UUID suffix.")

# --- Configuration ---
# NOTE: It's best practice to write to a NEW file to preserve your original data.
input_file = "synthetic_soap_notes.jsonl"
output_file = "synthetic_soap_notes_unique_names.jsonl"
# ---------------------

# Execute the function
make_client_names_individually_unique(input_file, output_file)