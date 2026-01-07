import os

# --- CONFIGURATION ---
# Name of the output file
OUTPUT_FILE = "codebase_context.txt"

# Folders to completely ignore (skips traversing them)
IGNORED_DIRS = {
    '.git', '.vscode', '.idea', '__pycache__', 'node_modules', 
    'build', 'dist', 'bin', 'obj', 'lib', 'include', 'vendor'
}

# File extensions to include (add or remove as needed)
# If you want EVERYTHING that is text, you can comment this out and use the binary check.
INCLUDED_EXTENSIONS = {
    # C/C++
    '.cpp', '.c', '.h', '.hpp', '.cc', '.hh', '.cxx', '.hxx', '.cmake', 'CMakeLists.txt',
    # Python
    '.py',
    # Web/Config
    '.json', '.yaml', '.yml', '.xml', '.md', '.txt',
    # Scripts
    '.sh', '.bat'
}

def is_binary(file_path):
    """
    Simple heuristic to check if a file is binary.
    Reads a small chunk and checks for null bytes.
    """
    try:
        with open(file_path, 'rb') as f:
            chunk = f.read(1024)
            return b'\0' in chunk
    except Exception:
        return True

def main():
    # Delete existing output file if it exists so we don't append to it
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
        # Walk through the current directory
        for root, dirs, files in os.walk("."):
            # 1. Modify 'dirs' in-place to skip ignored directories
            #    (This prevents os.walk from even entering them)
            dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]

            for file in files:
                file_path = os.path.join(root, file)
                
                # Check extension
                # (Remove this check if you want to rely solely on the binary check)
                _, ext = os.path.splitext(file)
                if ext not in INCLUDED_EXTENSIONS and file not in INCLUDED_EXTENSIONS:
                    continue

                # Skip the output file itself if it appears in the list
                if file == OUTPUT_FILE or file == "codebase_to_text.py":
                    continue

                # 2. Safety check for binary files (double check)
                if is_binary(file_path):
                    print(f"Skipping binary file: {file_path}")
                    continue

                # 3. Write to the output file
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as in_f:
                        content = in_f.read()
                        
                        # Write a clear header for NotebookLM
                        out_f.write(f"\n{'='*50}\n")
                        out_f.write(f"FILE PATH: {file_path}\n")
                        out_f.write(f"{'='*50}\n\n")
                        
                        out_f.write(content)
                        out_f.write("\n") # Ensure separation
                        
                    print(f"Added: {file_path}")

                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    print(f"\n--- DONE ---\nOutput saved to: {os.path.abspath(OUTPUT_FILE)}")

if __name__ == "__main__":
    main()