import json
import os
import time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_PDF_DIR = DATA_DIR / "raw_pdfs"
PROCESSED_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"

REQUEST_DELAY_SECONDS = 3
REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRIES = 3
MIN_VALID_PDF_BYTES = 2048
USER_AGENT = "AIStudyMate-CorpusDownloader/1.0 (academic research project)"

PDF_SOURCES = [
    {
        "id": "attention_is_all_you_need",
        "title": "Attention Is All You Need",
        "authors": "Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin",
        "year": 2017,
        "source_org": "arXiv (cs.CL)",
        "identifier": "arXiv:1706.03762",
        "url": "https://arxiv.org/pdf/1706.03762",
        "filename": "attention_is_all_you_need.pdf",
        "topic": "Attention mechanisms, Transformers",
    },
    {
        "id": "adam_optimizer",
        "title": "Adam: A Method for Stochastic Optimization",
        "authors": "Kingma, Ba",
        "year": 2014,
        "source_org": "arXiv (cs.LG)",
        "identifier": "arXiv:1412.6980",
        "url": "https://arxiv.org/pdf/1412.6980",
        "filename": "adam_optimizer.pdf",
        "topic": "Optimization, gradient descent, Adam",
    },
    {
        "id": "dropout",
        "title": "Dropout: A Simple Way to Prevent Neural Networks from Overfitting",
        "authors": "Srivastava, Hinton, Krizhevsky, Sutskever, Salakhutdinov",
        "year": 2014,
        "source_org": "JMLR 15(56):1929-1958",
        "identifier": "JMLR volume15/srivastava14a",
        "url": "http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf",
        "filename": "dropout.pdf",
        "topic": "Regularization, overfitting",
    },
    {
        "id": "batch_normalization",
        "title": "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift",
        "authors": "Ioffe, Szegedy",
        "year": 2015,
        "source_org": "arXiv (cs.LG)",
        "identifier": "arXiv:1502.03167",
        "url": "https://arxiv.org/pdf/1502.03167",
        "filename": "batch_normalization.pdf",
        "topic": "Regularization, training stability",
    },
    {
        "id": "resnet",
        "title": "Deep Residual Learning for Image Recognition",
        "authors": "He, Zhang, Ren, Sun",
        "year": 2015,
        "source_org": "arXiv (cs.CV)",
        "identifier": "arXiv:1512.03385",
        "url": "https://arxiv.org/pdf/1512.03385",
        "filename": "resnet.pdf",
        "topic": "CNN architectures, deep network training",
    },
    {
        "id": "lstm_search_space_odyssey",
        "title": "LSTM: A Search Space Odyssey",
        "authors": "Greff, Srivastava, Koutnik, Steunebrink, Schmidhuber",
        "year": 2015,
        "source_org": "arXiv (cs.NE)",
        "identifier": "arXiv:1503.04069",
        "url": "https://arxiv.org/pdf/1503.04069",
        "filename": "lstm_search_space_odyssey.pdf",
        "topic": "RNNs, LSTM, gates, vanishing gradients",
    },
    {
        "id": "gan",
        "title": "Generative Adversarial Networks",
        "authors": "Goodfellow, Pouget-Abadie, Mirza, Xu, Warde-Farley, Ozair, Courville, Bengio",
        "year": 2014,
        "source_org": "arXiv (stat.ML)",
        "identifier": "arXiv:1406.2661",
        "url": "https://arxiv.org/pdf/1406.2661",
        "filename": "gan.pdf",
        "topic": "Generative models, GANs",
    },
    {
        "id": "vae",
        "title": "Auto-Encoding Variational Bayes",
        "authors": "Kingma, Welling",
        "year": 2013,
        "source_org": "arXiv (stat.ML)",
        "identifier": "arXiv:1312.6114",
        "url": "https://arxiv.org/pdf/1312.6114",
        "filename": "vae.pdf",
        "topic": "Autoencoders, VAEs, latent variable models",
    },
    {
        "id": "ddpm",
        "title": "Denoising Diffusion Probabilistic Models",
        "authors": "Ho, Jain, Abbeel",
        "year": 2020,
        "source_org": "arXiv (cs.LG)",
        "identifier": "arXiv:2006.11239",
        "url": "https://arxiv.org/pdf/2006.11239",
        "filename": "ddpm.pdf",
        "topic": "Diffusion models, generative modeling",
    },
    {
        "id": "random_forests",
        "title": "Random Forests",
        "authors": "Breiman",
        "year": 2001,
        "source_org": "UC Berkeley Statistics",
        "identifier": "stat.berkeley.edu/~breiman/randomforest2001.pdf",
        "url": "https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf",
        "filename": "random_forests.pdf",
        "topic": "Ensemble methods, decision trees, random forests",
    },
    {
        "id": "feature_selection",
        "title": "An Introduction to Variable and Feature Selection",
        "authors": "Guyon, Elisseeff",
        "year": 2003,
        "source_org": "JMLR 3:1157-1182",
        "identifier": "JMLR volume3/guyon03a",
        "url": "http://www.jmlr.org/papers/volume3/guyon03a/guyon03a.pdf",
        "filename": "feature_selection.pdf",
        "topic": "Feature selection, feature engineering",
    },
    {
        "id": "knn_classifiers_tutorial",
        "title": "k-Nearest Neighbour Classifiers: 2nd Edition (with Python examples)",
        "authors": "Cunningham, Delany",
        "year": 2020,
        "source_org": "arXiv (cs.LG)",
        "identifier": "arXiv:2004.04523",
        "url": "https://arxiv.org/pdf/2004.04523",
        "filename": "knn_classifiers_tutorial.pdf",
        "topic": "k-Nearest Neighbors classification",
    },
    {
        "id": "cs229_notes1_supervised_learning",
        "title": "CS229 Lecture Notes - Supervised Learning (Linear Regression, GLMs)",
        "authors": "Ng et al.",
        "year": 2020,
        "source_org": "Stanford CS229",
        "identifier": "cs229-notes1.pdf",
        "url": "https://cs229.stanford.edu/summer2020/cs229-notes1.pdf",
        "filename": "cs229_notes1_supervised_learning.pdf",
        "topic": "Linear regression, GLMs, supervised learning fundamentals",
    },
    {
        "id": "cs229_notes2_generative_learning",
        "title": "CS229 Lecture Notes - Generative Learning Algorithms (GDA, Naive Bayes)",
        "authors": "Ng et al.",
        "year": 2020,
        "source_org": "Stanford CS229",
        "identifier": "cs229-notes2.pdf",
        "url": "https://cs229.stanford.edu/summer2020/cs229-notes2.pdf",
        "filename": "cs229_notes2_generative_learning.pdf",
        "topic": "Gaussian Discriminant Analysis, Naive Bayes",
    },
    {
        "id": "cs229_notes3_svm",
        "title": "CS229 Lecture Notes - Support Vector Machines",
        "authors": "Ng et al.",
        "year": 2020,
        "source_org": "Stanford CS229",
        "identifier": "cs229-notes3.pdf",
        "url": "https://cs229.stanford.edu/summer2020/cs229-notes3.pdf",
        "filename": "cs229_notes3_svm.pdf",
        "topic": "SVMs, kernels, classification",
    },
    {
        "id": "cs229_notes7a_unsupervised",
        "title": "CS229 Lecture Notes - Unsupervised Learning (k-means / EM)",
        "authors": "Ng et al.",
        "year": 2020,
        "source_org": "Stanford CS229",
        "identifier": "cs229-notes7a.pdf",
        "url": "https://cs229.stanford.edu/summer2020/cs229-notes7a.pdf",
        "filename": "cs229_notes7a_unsupervised.pdf",
        "topic": "Clustering, k-means, EM",
    },
    {
        "id": "cs229_bias_variance",
        "title": "CS229 Bias-Variance Analysis",
        "authors": "Stanford CS229 staff",
        "year": 2020,
        "source_org": "Stanford CS229",
        "identifier": "BiasVarianceAnalysis.pdf",
        "url": "https://cs229.stanford.edu/summer2020/BiasVarianceAnalysis.pdf",
        "filename": "cs229_bias_variance.pdf",
        "topic": "Overfitting, underfitting, bias-variance tradeoff",
    },
    {
        "id": "cs229_notes_deep_learning",
        "title": "CS229 Lecture Notes - Deep Learning (Neural Networks, Vectorization, Backpropagation)",
        "authors": "Ng, Katanforoosh, Avati",
        "year": 2020,
        "source_org": "Stanford CS229",
        "identifier": "cs229-notes-deep_learning.pdf",
        "url": "https://cs229.stanford.edu/summer2020/cs229-notes-deep_learning.pdf",
        "filename": "cs229_notes_deep_learning.pdf",
        "topic": "Neural network architecture, forward propagation, vectorization, backpropagation",
    },
    {
        "id": "cs229_evaluation_metrics",
        "title": "CS229 Section Notes - Evaluation Metrics (Classifiers)",
        "authors": "Chen, Avati",
        "year": 2020,
        "source_org": "Stanford CS229",
        "identifier": "evaluation_metrics_spring2020.pdf",
        "url": "https://cs229.stanford.edu/section/evaluation_metrics_spring2020.pdf",
        "filename": "cs229_evaluation_metrics.pdf",
        "topic": "Confusion matrix, accuracy, precision, recall, F1-score, ROC-AUC",
    },
    {
        "id": "cs231n_cnn_notes",
        "title": "CS231n Convolutional Neural Networks for Visual Recognition - Module 1 Notes",
        "authors": "Fei-Fei Li, Karpathy, Johnson",
        "year": 2021,
        "source_org": "Stanford CS231n",
        "identifier": "cs231n_cnn_notes",
        "url": "https://cs231n.stanford.edu/slides/2021/lecture_5.pdf",
        "filename": "cs231n_cnn_architectures.pdf",
        "topic": "Convolutional Neural Networks, spatial structure, parameter sharing, pooling",
    },
    {
        "id": "cs224n_rnn_notes",
        "title": "CS224n: Natural Language Processing with Deep Learning - Recurrent Neural Networks & LSTMs",
        "authors": "Stanford CS224n Staff",
        "year": 2019,
        "source_org": "Stanford CS224n",
        "identifier": "cs224n_rnn_notes",
        "url": "https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/readings/cs224n-2019-notes05-LM_RNN.pdf",
        "filename": "cs224n_rnn_and_lstm.pdf",
        "topic": "Recurrent Neural Networks, sequential dependencies, vanishing/exploding gradients, LSTMs",
    },
    {
        "id": "stanford_cs230_deep_learning_cheatsheet",
        "title": "Stanford CS230 Deep Learning VIP Cheatsheet",
        "authors": "Afshine Amidi, Shervine Amidi",
        "year": 2020,
        "source_org": "Stanford CS230 / GitHub",
        "identifier": "cs230_cheatsheet",
        "url": "https://raw.githubusercontent.com/afshinea/stanford-cs-230-deep-learning/master/en/super-cheatsheet-deep-learning.pdf",
        "filename": "cs230_deep_learning_cheatsheet.pdf",
        "topic": "Convolutional networks, Recurrent networks, Autoencoders, Optimization, Regularization",
    },
    {
        "id": "stanford_cs229_machine_learning_cheatsheet",
        "title": "Stanford CS229 Machine Learning VIP Cheatsheet",
        "authors": "Afshine Amidi, Shervine Amidi",
        "year": 2020,
        "source_org": "Stanford CS229 / GitHub",
        "identifier": "cs229_cheatsheet",
        "url": "https://raw.githubusercontent.com/afshinea/stanford-cs-229-machine-learning/master/en/super-cheatsheet-machine-learning.pdf",
        "filename": "cs229_machine_learning_cheatsheet.pdf",
        "topic": "Supervised, Unsupervised learning, Decision trees, SVMs, Metrics, Regularization",
    },
    {
        "id": "autoencoders_tutorial",
        "title": "Unsupervised Feature Learning and Deep Learning: Autoencoders",
        "authors": "Andrew Ng et al.",
        "year": 2022,
        "source_org": "arXiv (cs.LG)",
        "identifier": "autoencoders_intro",
        "url": "https://arxiv.org/pdf/2201.03898.pdf",
        "filename": "autoencoders_intro.pdf",
        "topic": "Autoencoders, dimensionality reduction, feature extraction, latent representation",
    },
]

DOCUMENTED_ONLY_SOURCES = [
    {
        "id": "backpropagation_rumelhart_1986",
        "title": "Learning representations by back-propagating errors",
        "authors": "Rumelhart, Hinton, Williams",
        "year": 1986,
        "source_org": "Nature, Volume 323, pages 533-536",
        "identifier": "DOI:10.1038/323533a0",
        "access_note": "Scanned image without embedded text layer. Superseded by cs229_notes_deep_learning.",
        "topic": "Backpropagation, neural network training, error propagation",
    },
    {
        "id": "adaboost",
        "title": "A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting",
        "authors": "Freund, Schapire",
        "year": 1997,
        "source_org": "Journal of Computer and System Sciences 55(1):119-139",
        "identifier": "DOI:10.1006/jcss.1997.1504",
        "access_note": "No stable open-access PDF mirror verified.",
        "topic": "Ensemble methods, boosting",
    },
    {
        "id": "deep_learning_book",
        "title": "Deep Learning",
        "authors": "Goodfellow, Bengio, Courville",
        "year": 2016,
        "source_org": "MIT Press",
        "identifier": "https://www.deeplearningbook.org/",
        "access_note": "Free HTML web edition, not a single downloadable PDF.",
        "topic": "Full DL foundations",
    },
]

SUPPLEMENTARY_DOC_SOURCES = [
    {
        "id": "sklearn_tree",
        "title": "scikit-learn User Guide - Decision Trees",
        "source_org": "scikit-learn official documentation",
        "url": "https://scikit-learn.org/stable/modules/tree.html",
        "topic": "Decision Trees",
    },
    {
        "id": "sklearn_neighbors",
        "title": "scikit-learn User Guide - Nearest Neighbors",
        "source_org": "scikit-learn official documentation",
        "url": "https://scikit-learn.org/stable/modules/neighbors.html",
        "topic": "k-Nearest Neighbors",
    },
    {
        "id": "sklearn_naive_bayes",
        "title": "scikit-learn User Guide - Naive Bayes",
        "source_org": "scikit-learn official documentation",
        "url": "https://scikit-learn.org/stable/modules/naive_bayes.html",
        "topic": "Naive Bayes",
    },
    {
        "id": "sklearn_clustering",
        "title": "scikit-learn User Guide - Clustering",
        "source_org": "scikit-learn official documentation",
        "url": "https://scikit-learn.org/stable/modules/clustering.html",
        "topic": "k-Means, clustering evaluation",
    },
    {
        "id": "sklearn_model_evaluation",
        "title": "scikit-learn User Guide - Model Evaluation",
        "source_org": "scikit-learn official documentation",
        "url": "https://scikit-learn.org/stable/modules/model_evaluation.html",
        "topic": "Metrics, cross-validation",
    },
]


def ensure_directories():
    for directory in (RAW_PDF_DIR, PROCESSED_DIR, METADATA_DIR, VECTOR_STORE_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def validate_pdf(path):
    if not path.exists():
        return False
    if path.stat().st_size < MIN_VALID_PDF_BYTES:
        return False
    with open(path, "rb") as f:
        header = f.read(5)
    return header == b"%PDF-"


def download_file(url, dest_path):
    headers = {"User-Agent": USER_AGENT}
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            request = Request(url, headers=headers)
            with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                content = response.read()
            with open(dest_path, "wb") as f:
                f.write(content)
            return True, None
        except (HTTPError, URLError, TimeoutError, OSError) as error:
            last_error = str(error)
            if attempt < MAX_RETRIES:
                time.sleep(2 * attempt)
    return False, last_error


def process_pdf_sources():
    ensure_directories()
    results = []

    print(f"Starting download process for {len(PDF_SOURCES)} resources...")

    for source in PDF_SOURCES:
        dest_path = RAW_PDF_DIR / source["filename"]
        record = dict(source)
        record["local_path"] = str(dest_path.relative_to(PROJECT_ROOT))

        if validate_pdf(dest_path):
            record["status"] = "skipped_existing"
            record["error"] = None
            print(f"Skipped (already exists): {source['filename']}")
        else:
            print(f"Downloading: {source['title']}...")
            success, error = download_file(source["url"], dest_path)

            if success and validate_pdf(dest_path):
                record["status"] = "downloaded"
                record["error"] = None
                print(f"Saved successfully: {source['filename']}")
            else:
                record["status"] = "failed"
                record["error"] = error or "Invalid PDF binary structure"
                print(f"Failed: {source['filename']} | Error: {record['error']}")

            time.sleep(REQUEST_DELAY_SECONDS)

        results.append(record)

    metadata_file = METADATA_DIR / "pdf_sources_manifest.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"Manifest saved to {metadata_file}")
    return results


if __name__ == "__main__":
    process_pdf_sources()