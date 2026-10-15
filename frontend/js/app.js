const API_BASE =
    "https://your-api-url.com";

async function callApi(
    endpoint
) {

    try {

        const response =
            await fetch(
                API_BASE +
                endpoint,
                {
                    method: "POST"
                }
            );

        const data =
            await response.json();

        document
            .getElementById(
                "output"
            )
            .textContent =
            JSON.stringify(
                data,
                null,
                2
            );

    } catch (error) {

        document
            .getElementById(
                "output"
            )
            .textContent =
            error.toString();
    }
}

function alignPages() {

    callApi(
        "/align"
    );
}

function generateDrafts() {

    callApi(
        "/drafts"
    );
}

function buildDataset() {

    callApi(
        "/dataset"
    );
}

function trainModel() {

    callApi(
        "/train"
    );
}
