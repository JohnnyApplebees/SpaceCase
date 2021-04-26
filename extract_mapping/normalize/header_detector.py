import pandas as pd
import boto3
from s3path import S3Path
import argparse
# Download spacy model if hasn't been
#!python -m spacy download en_core_web_lg

# Import custom modules
import nlp_proc, organize

# Command line argument - UUID
parser = argparse.ArgumentParser(
    description='This script assigns headers to textract and detectron extracts and saves post-processed versions.'
)

parser.add_argument('--uuid', required=True, metavar='uuid', help='UUID of extracted PDF')
args = parser.parse_args()
UUID = args.uuid
print(UUID)

# Method tp perform header detection and assignment
def get_header(path):
    # Process documents
    if 'detectron' in str(path):
        df = pd.read_csv("s3:/" + str(path)).iloc[:, 1:] # For detectron, drop first (index) column
        df = organize.extra_row_remover(df)
        # Use shape of table heuristic to determine if the table is useful or not
        # if row*col count <=15, classify table as header or other information
        # Analysis was conducted and documented in Damian Doc Extraction Whiteboard
        if df.shape[0] * df.shape[1] <= 15:
            # TODO: document that this table was labelled as extraneous somehow
            print(f'Table: {path} contains minimal data, not processed')
        else:
            df_detectron = nlp_proc.header_detector(df, mf_rr_kb).replace_header()
            #_,zscores = nlp_proc.header_detector(df, mf_rr_kb).get_header_candidates()
            #zscores = pd.DataFrame(zscores, columns=['idx','v','z'])

            # Save output to S3
            df_detectron.to_csv(postproc_output_dir + "detectron/" + UUID + "/" + str(path).split('/')[-1])
            #zscores.to_csv(postproc_output_dir + "detectron/" + UUID + "/_zscores_" + str(path).split('/')[-1])

    elif 'textract' in str(path):
        with path.open() as f:
            content = f.readlines()
        index = [x for x in range(len(content)) if 'Table' in content[x]]
        if len(index) > 1:
            row_len = [len(x) for x in content]
            print(f"Table: {path} has multiple subtables, code in development")
        else:
                df = pd.read_csv("s3:/" + str(path), skiprows=1, header=None)
                df = organize.extra_row_remover(df)
                df_textract = nlp_proc.header_detector(df, mf_rr_kb).replace_header()
                #_,zscores = nlp_proc.header_detector(df, mf_rr_kb).get_header_candidates()
                #zscores = pd.DataFrame(zscores, columns=['idx','v','z'])
                
                # Save output to S3
                df_textract.to_csv(postproc_output_dir + "textract/" + UUID + "/" + str(path).split('/')[-1])
                #zscores.to_csv(postproc_output_dir + "textract/" + UUID + "/_zscores_" + str(path).split('/')[-1])
    else:
        print('Extraction source not recognized')

# parameters should be UUID
#UUID = 'f3d3fe84-a2ca-11eb-9113-666251992ff6'
postproc_output_dir = "s3://tab-data-extraction-sandbox/postproc_output/"

# Read in Multifamily header knowledge base
mf_rr_kb = pd.read_csv('s3://tab-data-extraction-sandbox/manual_review/rr_multifamily_header.csv')

# Get list of paths of all CSVs given UUID
rr_path = S3Path('/dataingest-pdfextraction-output/')
detectron_paths = list(rr_path.glob('detectron_output/' + UUID + '/*.csv'))
textract_paths = list(rr_path.glob('textract_output/' + UUID + '.pdf-analysis/*tables.csv'))

# Run header_separator
for p in detectron_paths:
    get_header(p)

for p in textract_paths:
    get_header(p)