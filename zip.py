import shutil

# path of the folder to be zipped

folder_path='visualize'

# name of the output zip file(without.zip extension)

output_zip_path='results'

shutil.make_archive(output_zip_path, 'zip', folder_path)

print(f"Successfully created {output_zip_path}.zip")
