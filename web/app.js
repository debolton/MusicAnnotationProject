const API =
    "http://localhost:8000";

async function callApi(
    endpoint
) {

    const response =
        await fetch(
            API + endpoint,
            {
                method: "POST"
            }
        );

    const data =
        await response.json();

    document.getElementById(
        "output"
    ).textContent =
        JSON.stringify(
            data,
            null,
            2
        );
}

function align() {

    callApi("/align");
}

function buildDataset() {

    callApi("/dataset");
}

function trainModel() {

    callApi("/train");
}
