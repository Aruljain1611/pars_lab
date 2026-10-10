from flask import Flask, render_template, request

app = Flask(__name__)

datasets = {
    "mimic": {
        "name": "MIMIC-IV",
        "authors": "Johnson, A. E. W., Bulgarelli, L., Shen, L., et al.",
        "year": "2021",
        "version": "Version 1.0",
        "publisher": "PhysioNet",
        "doi": "https://doi.org/10.13026/s6n6-xd98"
    },

    "california": {
        "name": "California Housing Dataset",
        "authors": "Pace, R. K., & Barry, R.",
        "year": "1997",
        "publisher": "California Census Data / scikit-learn",
        "url": "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html"
    }
}


def generate_citation(dataset):
    if "doi" in dataset:
        return (
            f"{dataset['authors']} ({dataset['year']}). "
            f"{dataset['name']} ({dataset['version']}) [Dataset]. "
            f"{dataset['publisher']}. {dataset['doi']}"
        )

    return (
        f"{dataset['authors']} ({dataset['year']}). "
        f"{dataset['name']} [Dataset]. "
        f"{dataset['publisher']}. {dataset['url']}"
    )


@app.route("/", methods=["GET", "POST"])
def index():
    citation = None
    selected_dataset = None

    if request.method == "POST":
        selected_dataset = request.form.get("dataset")

        if selected_dataset in datasets:
            citation = generate_citation(datasets[selected_dataset])

    return render_template(
        "index.html",
        datasets=datasets,
        citation=citation,
        selected_dataset=selected_dataset
    )


if __name__ == "__main__":
    app.run(debug=True)