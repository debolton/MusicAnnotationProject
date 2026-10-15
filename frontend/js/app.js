const API_BASE =
    "https://your-api-url.com";

async function callApi(endpoint) {

    const output =
        document.getElementById(
            "output"
        );

    output.textContent =
        "Running...";

    try {

        const response =
            await fetch(
                API_BASE + endpoint,
                {
                    method: "POST"
                }
            );

        const data =
            await response.json();

        output.textContent =
            JSON.stringify(
                data,
                null,
                2
            );

    }
    catch (error) {

        output.textContent =
            "ERROR\n\n" + error;
    }
}

function alignPages() {
    callApi("/align");
}

function generateDrafts() {
    callApi("/drafts");
}

function buildDataset() {
    callApi("/dataset");
}

function validateRelease() {
    callApi("/validate");
}

function trainModel() {
    callApi("/train");
}
