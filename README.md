# Κ23γ: Ανάπτυξη Λογισμικού για Αλγοριθμικά Προβλήματα Χειμερινό Εξάμηνο 2025-26
## 3η Προγραμματιστική Εργασία: Αναζήτηση "Απομακρυσμένων" Ομολόγων με Προσεγγιστικές Μεθόδους & ESM-2
### Ομάδα Χρηστών 3
Αλατζάς Αλέξανδρος - 1115201900005  
Σκλιάμης Ιωάννης - 1115201700141

### Κατάλογος αρχείων και περιγραφή
```
root
│   README.md                # Οδηγίες χρήσης και περιγραφή
│   Makefile                 # Αυτόματη μεταγλώττιση 1ης εργασίας
|   Analysis.md              # Αναφορά βιολογικής αξιολόγησης
|   algo_runner.py           # Εκτέλεση κωδίκων προηγούμενων εργασιών (optimal params)
|   analysis_script.py       # Παραγωγή γραφημάτων Recall vs QPS
│   args_parser.py           # Ανάγνωση και validation παραμέτρων
│   dataset.py               # Κλάσεις αναπαράστασης των dataset
│   feedforward_nn.py        # Υλοποίηση feedforward νευρωνικού
│   graph_builder.py         # Συναρτήσεις κατασκευής knn γράφου
│   nlsh_build.py            # Κατασκευή ευρετηρίου για nlsh
│   nlsh_search.py           # Αναζήτηση ευρετηρίου για nlsh
│   read_dataset.py          # Συναρτήσεις ανάγνωσης των dataset
│   requirements.txt         # Απαραίτητες βιβλιοθήκες
│   set_seed.py              # Συνάρτηση ανάθεσης seed
│   train.py                 # Συνάρτηση εκπαίδευσης νεωρωνικού
│   protein_embed.py         # Παραγωγή embeddings
│   protein_search.py        # Αναζήτηση πρωτεϊνών
|   results.txt              # Αρχείο εξόδου για Ν=20
└───include/
│   │   Dataset.hpp          # Δηλώσεις της κλάσης αναπαράστασης Dataset
│   │   euclidean.hpp        # Δήλωση συνάρτησης Ευκλείδιας απόστασης
│   │   helperStructs.hpp    # Δηλώσεις κοινών structs για όλους τους αλγόριθμους
│   │   Hypercube.hpp        # Δηλώσεις της κλάσης του αλγορίθμου Hypercube
│   │   ivfbase.hpp          # Δηλώσεις της κλάσης του αλγορίθμου συσταδοποίησης KMeans
│   │   ivfflat.hpp          # Δηλώσεις της κλάσης του αλγορίθμου IVFFLAT
│   │   ivfpq.hpp            # Δηλώσεις της κλάσης του αλγορίθμου IVFPQ
│   │   LSH.hpp              # Δηλώσεις της κλάσης του αλγορίθμου LSH
│   │   parsingFuncs.hpp     # Δηλώσεις συναρτήσεων για την ανάγνωση παραμέτρων
└───src/
│   │   Dataset.cpp          # Υλοποίηση κλάσης αναπαράστασης Dataset
│   │   euclidean.cpp        # Υλοποίηση συνάρτησης Ευκλείδιας απόστασης
│   │   Hypercube.cpp        # Υλοποίηση κλάσης του αλγορίθμου Hypercube
│   │   ivfbase.cpp          # Υλοποίηση κλάσης του αλγορίθμου συσταδοποίησης KMeans
│   │   ivfflat.cpp          # Υλοποίηση κλάσης του αλγορίθμου IVFFLAT
│   │   ivfpq.cpp            # Υλοποίηση κλάσης του αλγορίθμου IVFPQ
│   │   LSH.cpp              # Υλοποίηση κλάσης του αλγορίθμου LSH
│   │   main.cpp             # Η main συνάρτηση
│   │   parsingFuncs.hpp     # Υλοποίηση συναρτήσεων για την ανάγνωση παραμέτρων
└───experiments/             # Αρχεία πειραμάτων
│   │   lsh_experiments_results1.csv
│   │   lsh_experiments_results2.csv
│   │   hypercube_experiments_results.csv
│   │   ivfflat_experiments_results.csv
│   │   ivfpq_experiments_results.csv
│   │   neural_experiments_results.csv
└───images/                  # Γραφήματα απόδοσης
│   │   Euclidean_LSH_recall_vs_qps.png
│   │   Hypercube_recall_vs_qps.png
│   │   IVFFlat_recall_vs_qps.png
│   │   IVFPQ_recall_vs_qps.png
│   │   Neural_LSH_recall_vs_qps.png
└──────────────────────────
```

### Οδηγίες εγκατάστασης εξαρτήσεων
Για το project χρειαστήκαμε τις βιβλιοθήκες της 2ης εργασίας: torch, torchsummary,kahip και επιπλέον τις fair-esm και tabulate.  
Βρίσκονται στο αρχείο requirements.txt και εγκαθίστανται με την εντολή pip install -r requirements.txt.  
Η υλοποίηση έγινε σε σύστημα Linux και Python 3.13.5 χρησιμοποιώντας το περιβάλλον Anaconda με όλες τις προεγκατεστημένες βιβλιοθήκες (numpy, scikit-learn, matplotlib).  
Για την εκτέλεση του BLAST, απαιτείται η εντολή `sudo apt install ncbi-blast+`.

### Οδηγίες χρήσης
Τα δύο σενάρια εκτελούνται σύμφωνα με τις απαιτήσεις της εκφώνησης. Η εκτέλεση του κάθε σεναρίου γίνεται ως εξής:
- `$python3 protein_embed.py -i swissprot_50k.fasta -o protein_vectors.bin`
- `$python3 protein_search.py -d protein_vectors.bin -q targets.fasta -o output.txt -method <all|lsh|hypercube|neural|ivfflat|ivfpq>`

Θεωρούμε ότι τα datasets βρίσκονται στο root directory της εργασίας.

### Σχεδιαστικές επιλογές και παραδοχές

#### read_dataset.py
`read_fasta()`: Ανάγνωση του αρχείου FASTA. Επιστρέφει λίστα από (sequence_id, sequence) tuples.  
`read_embeddings()`: Φορτώνει το binary αρχείο με τα protein embeddings.
`read_protein_id_map()`: Διαβάζει αρχείο κειμένου όπου κάθε γραμμή περιέχει ID πρωτεΐνης και τα αντιστοιχίζει στα IDs του αρχικού dataset.
`read_query_id_map()`: Διαβάζει query αρχείο FASTA και αντιστοιχίζει ID με ακολουθίες πρωτεΐνης.
`read_blast_tsv()`: Διαβάζει το `tsv` αρχείο εξόδου του BLAST και αντιστοιχίζει τα query IDs στα protein hits.
`read_blast_identities()`: Διαβάζει το `tsv` αρχείο εξόδου του BLAST και εξάγει τις τιμές του Identity για κάθε query.

#### protein_embed.py
Αρχικά, δημιουργούνται τα embeddings των πρωτεϊνών χρησιμοποιώντας το προεκπαιδευμένο μοντέλο ESM-2. Διαβάζουμε τις ακολουθίες πρωτεϊνών από το αρχείο
FASTA, τις επεξεργαζόμαστε μέσω του μοντέλου, εφαρμόζουμε mean pooling για την παραγωγή διανυσμάτων 320 διαστάσεων και αποθηκεύουμε τόσο τα embeddings όσο και τα αντίστοιχα IDs των πρωτεϊνών για την αναζήτηση σε επόμενο βήμα.

#### readFuncs.cpp
`readTSV()`: Διαβάζει το `tsv` αρχείο εξόδου του BLAST και αντιστοιχίζει τα query IDs στα protein hits  
`readProteinIdMap()`: Διαβάζει αρχείο κειμένου όπου κάθε γραμμή περιέχει ID πρωτεΐνης και τα αντιστοιχίζει στα IDs του αρχικού dataset.  
`readQueryIdMap()`: Διαβάζει το αρχείο targets.fasta και αντιστοιχίζει τα query protein IDs στα IDs του query dataset.  
`readIdentities()`: Διαβάζει το `tsv` αρχείο εξόδου του BLAST και αντιστοιχίζει τα queries με τα protein hits και το % identity.

#### Αλλαγές σε προηγούμενα παραδοτέα (LSH, Hypercube, IVFFlat, IVFPQ, NLSH)
Για την 3η εργασία, αλλάξαμε τα σημεία στα οποία γίνεται η αναζήτηση, ώστε να προσαρμοστεί στο νέο ζητούμενο.  
Δεν χρησιμοποιούμε πλέον εξαντλητική αναζήτηση, αλλά τα πραγματικά BLAST hits και αντιστοίχως υπολογίζουμε το Recall@N σύμφωνα με τα BLAST top-N hits και όχι με τις ευκλείδιες αποστάσεις.

#### algo_runner.py
Βοηθητικό script για την εκτέλεση των ANN αλγορίθμων των προηγούμενων εργασιών. Αρχικά, εκτελείται make και έπειτα μέσω της subprocess εκτελείται ο αλγόριθμος (ή όλοι) που δόθηκε από τον χρήστη. Οι παράμετροι εκτέλεσης βρίσκονται hard-coded στην αρχή του script και ορίστηκαν από εμάς ως βέλτιστες για το συγκεκριμένο σενάριο χρήσης, δηλαδή με το υψηλότερο Recall.

#### protein_search.py
Όμοια με τα embeddings για το αρχείο πρωτεϊνών, δημιουργούμε τα embeddings για τις query πρωτεΐνες, μέσω του ESM-2. Διαβάζουμε τις ακολουθίες πρωτεϊνών από το αρχείο
FASTA, τις επεξεργαζόμαστε μέσω του μοντέλου, εφαρμόζουμε mean pooling για την παραγωγή διανυσμάτων 320 διαστάσεων και τα αποθηκεύουμε. Χρησιμοποιώντας τη subprocess, δημιουργούμε τη βάση δεδομένων BLAST μέσω `makeblastdb` και εκτελούμε αναζήτηση στις query πρωτεΐνες. Ο χρόνος εκτέλεσης καταγράφεται ώστε να συγκριθεί με τις ANN μεθόδους. Το `algo_runner` εκτελεί τον αλγόριθμο που δόθηκε από τον χρήστη χρησιμοποιώντας τα παραχθέντα embeddings. Τέλος, συλλέγονται τα αποτελέσματα των ANN για όλα τα queries και εκτυπώνεται το συγκεντρωτικό report για 20 πλησιέστερους γείτονες ανά query. Το τελικό report γράφεται στο αρχείο που δόθηκε ως όρισμα στην εντολή εκτέλεσης.

#### print_results.py
Βοηθητικό script για την ζητούμενη παραγωγή εξόδου δύο επιπέδων. Εφόσον οι ANN αλγόριθμοι εκτελούνται από το script μας, κάνουμε την παραδοχή ότι γνωρίζουμε εκ των προτέρων τα ονόματα των output αρχείων τους, συνεπώς τα δίνουμε hard-coded στο script για να τα κάνουμε parse. Για την αναπαράσταση σε πίνακες, χρησιμοποιήθηκε η βιβλιοθήκη tabulate της Python. Για κάθε query πρωτεΐνη, το QPS της υπολογίζεται ως το αντίστροφο του χρόνου εκτέλεσης.

#### analysis_script.py
Βοηθητικό script στο οποίο διαβάζουμε τα αποτελέσματα των πειραμάτων μας από τα `csv` αρχεία και παράγουμε τις γραφικές παραστάσεις απόδοσης (Recall@N vs QPS). Περνάμε κάθε σύνολο πειραμάτων σε ένα dataframe και σχηματίζουμε το αντίστοιχο γράφημα.
