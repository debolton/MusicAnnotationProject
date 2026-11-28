const $ = x => document.getElementById(x);

let sid = null;
let mode = 'pan';
let zoom = 1;
let drawing = false;
let pts = [];
let panning = false;
let last = null;

const page = $('page');
const mask = $('mask');
const cv = $('draw');
const ctx = cv.getContext('2d');
const stage = $('stage');
const vp = $('viewport');


function status(message) {
    $('status').textContent = message;
}


function setMode(newMode) {
    mode = newMode;

    document.querySelectorAll('[data-mode]').forEach(
        button => {
            button.classList.toggle(
                'active',
                button.dataset.mode === newMode
            );
        }
    );
}


function applyZoom() {
    stage.style.transform = `scale(${zoom})`;

    stage.style.width =
        (page.naturalWidth * zoom) + 'px';

    stage.style.height =
        (page.naturalHeight * zoom) + 'px';
}


function fit() {
    if (!page.naturalWidth) {
        return;
    }

    zoom = Math.min(
        vp.clientWidth / page.naturalWidth,
        (vp.clientHeight - 4) / page.naturalHeight
    );

    applyZoom();

    vp.scrollTo(0, 0);
}


function describeError(data, fallback) {
    if (!data) {
        return fallback;
    }

    if (typeof data === 'string') {
        return data;
    }

    if (typeof data.detail === 'string') {
        return data.detail;
    }

    if (data.detail !== undefined) {
        try {
            return JSON.stringify(data.detail);
        } catch {
            return String(data.detail);
        }
    }

    if (data.message) {
        return String(data.message);
    }

    try {
        return JSON.stringify(data);
    } catch {
        return fallback;
    }
}


async function readResponse(response) {
    const contentType =
        response.headers.get('content-type') || '';

    if (
        contentType.includes('application/json')
    ) {
        return await response.json();
    }

    const text = await response.text();

    if (!text) {
        return {};
    }

    try {
        return JSON.parse(text);
    } catch {
        return {
            detail: text
        };
    }
}


function cacheBust(url) {
    if (!url) {
        return url;
    }

    return (
        url +
        (url.includes('?') ? '&' : '?') +
        't=' +
        Date.now()
    );
}


/* ============================================================
   SESSION CREATION / YOLO
   ============================================================ */

$('generate').onclick = async () => {
    const file = $('file').files[0];

    if (!file) {
        status('Select a page first.');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    status('Running YOLO annotation detector...');

    try {
        const response = await fetch(
            '/api/session',
            {
                method: 'POST',
                body: formData
            }
        );

        const data = await readResponse(
            response
        );

        if (!response.ok) {
            status(
                'YOLO/session creation failed: ' +
                describeError(
                    data,
                    `HTTP ${response.status}`
                )
            );
            return;
        }

        if (
            !data.session_id ||
            !data.image_url ||
            !data.overlay_url
        ) {
            status(
                'Session creation failed: server returned ' +
                'an incomplete response.'
            );
            return;
        }

        sid = data.session_id;

        page.onload = () => {
            cv.width = page.naturalWidth;
            cv.height = page.naturalHeight;

            cv.style.width =
                page.naturalWidth + 'px';

            cv.style.height =
                page.naturalHeight + 'px';

            fit();
        };

        page.src = cacheBust(
            data.image_url
        );

        mask.src = cacheBust(
            data.overlay_url
        );

        $('upload').hidden = true;
        $('review').hidden = false;

        status(
            'Review the authoritative YOLO mask ' +
            'with ADD / ERASE.'
        );

    } catch (error) {
        status(
            'YOLO/session creation failed: ' +
            error.message
        );
    }
};


/* ============================================================
   VIEW / EDIT CONTROLS
   ============================================================ */

document
    .querySelectorAll('[data-mode]')
    .forEach(
        button => {
            button.onclick = () =>
                setMode(
                    button.dataset.mode
                );
        }
    );


$('fit').onclick = fit;


$('zin').onclick = () => {
    zoom = Math.min(
        4,
        zoom * 1.25
    );

    applyZoom();
};


$('zout').onclick = () => {
    zoom = Math.max(
        0.15,
        zoom / 1.25
    );

    applyZoom();
};


function point(event) {
    const rect =
        cv.getBoundingClientRect();

    return [
        (
            event.clientX -
            rect.left
        ) * cv.width / rect.width,

        (
            event.clientY -
            rect.top
        ) * cv.height / rect.height
    ];
}


cv.onpointerdown = event => {
    event.preventDefault();

    cv.setPointerCapture(
        event.pointerId
    );

    if (mode === 'pan') {
        panning = true;

        last = [
            event.clientX,
            event.clientY
        ];

        return;
    }

    drawing = true;
    pts = [point(event)];
};


cv.onpointermove = event => {
    if (panning) {
        vp.scrollLeft -=
            event.clientX - last[0];

        vp.scrollTop -=
            event.clientY - last[1];

        last = [
            event.clientX,
            event.clientY
        ];

        return;
    }

    if (!drawing) {
        return;
    }

    pts.push(
        point(event)
    );

    ctx.clearRect(
        0,
        0,
        cv.width,
        cv.height
    );

    ctx.strokeStyle =
        mode === 'add'
            ? '#06f'
            : '#0c5';

    ctx.lineWidth =
        +$('brush').value;

    ctx.lineCap = 'round';

    ctx.beginPath();

    pts.forEach(
        (p, index) => {
            if (index) {
                ctx.lineTo(...p);
            } else {
                ctx.moveTo(...p);
            }
        }
    );

    ctx.stroke();
};


cv.onpointerup = async () => {
    if (panning) {
        panning = false;
        return;
    }

    if (!drawing) {
        return;
    }

    drawing = false;

    try {
        const response = await fetch(
            `/api/session/${sid}/stroke`,
            {
                method: 'POST',
                headers: {
                    'Content-Type':
                        'application/json'
                },
                body: JSON.stringify({
                    mode: mode,
                    brush: +$('brush').value,
                    points: pts,
                    view_width: cv.width,
                    view_height: cv.height
                })
            }
        );

        const data = await readResponse(
            response
        );

        ctx.clearRect(
            0,
            0,
            cv.width,
            cv.height
        );

        if (!response.ok) {
            status(
                'Mask edit failed: ' +
                describeError(
                    data,
                    `HTTP ${response.status}`
                )
            );
            return;
        }

        if (!data.overlay_url) {
            status(
                'Mask edit failed: server returned ' +
                'no overlay URL.'
            );
            return;
        }

        mask.src = cacheBust(
            data.overlay_url
        );

        status(
            `Mask revision ${data.revision} saved.`
        );

    } catch (error) {
        ctx.clearRect(
            0,
            0,
            cv.width,
            cv.height
        );

        status(
            'Mask edit failed: ' +
            error.message
        );
    }
};


/* ============================================================
   UNDO
   ============================================================ */

$('undo').onclick = async () => {
    try {
        const response = await fetch(
            `/api/session/${sid}/undo`,
            {
                method: 'POST'
            }
        );

        const data = await readResponse(
            response
        );

        if (!response.ok) {
            status(
                'Undo failed: ' +
                describeError(
                    data,
                    `HTTP ${response.status}`
                )
            );
            return;
        }

        if (!data.overlay_url) {
            status(
                'Undo failed: server returned ' +
                'no overlay URL.'
            );
            return;
        }

        mask.src = cacheBust(
            data.overlay_url
        );

        status(
            `Returned to mask revision ${data.revision}.`
        );

    } catch (error) {
        status(
            'Undo failed: ' +
            error.message
        );
    }
};


/* ============================================================
   AUTHORISE MASK / RUN V7.8 + A5-v1
   ============================================================ */

$('authorise').onclick = async () => {
    status(
        'Running image cleanup... '
    );

    /*
     * Any previously packaged ZIP belongs to the previous
     * correction/review state and must not remain exposed.
     */
    $('download').hidden = true;
    $('download').removeAttribute(
        'href'
    );

    try {
        const response = await fetch(
            `/api/session/${sid}/authorise`,
            {
                method: 'POST'
            }
        );

        const data = await readResponse(
            response
        );

        if (!response.ok) {
            status(
                'Correction failed: ' +
                describeError(
                    data,
                    `HTTP ${response.status}`
                )
            );
            return;
        }

        if (
            !data.original_url ||
            !data.corrected_url
        ) {
            status(
                'Correction failed: server returned ' +
                'incomplete result URLs.'
            );
            return;
        }

        $('whole').hidden = false;

        $('original').src =
            cacheBust(
                data.original_url
            );

        $('corrected').src =
            cacheBust(
                data.corrected_url
            );

        /*
         * A5-v1 decision diagnostics:
         *
         * PRINT      = green
         * ANNOTATION = red
         * UNCERTAIN  = yellow/amber
         *
         * V7.8 routing remains a separate evidence family.
         */
        const diagnostics = [
            [
                'decisions',
                'A5 decisions'
            ],
            [
                'v78',
                'V7.8 baseline'
            ],
            [
                'a5',
                'A5 corrected'
            ],
            [
                'delta',
                'A5 vs V7.8 difference'
            ],
            [
                'print',
                'PRINT mask'
            ],
            [
                'eligible_a5_v1',
                'A5 eligible'
            ],
            [
                'preserve',
                'V7.8 preserve'
            ],
            [
                'remove',
                'V7.8 remove'
            ],
            [
                'ambiguous',
                'V7.8 ambiguous'
            ],
            [
                'mask',
                'Prepared mask'
            ],
            [
                'telea',
                'Telea'
            ],
            [
                'paper',
                'Paper estimate'
            ]
        ];

        $('diaglinks').innerHTML =
            diagnostics
                .map(
                    ([key, label]) =>
                        `<a target="_blank" ` +
                        `rel="noopener" ` +
                        `href="/api/session/${sid}/diagnostic/${key}">` +
                        `${label}</a>`
                )
                .join(' · ');

        const d =
            data.diagnostics || {};

        const eligible =
            d.a5_eligible_pixels ?? '?';

        const printPixels =
            d.a5_print_pixels ?? '?';

        const changed =
            d.a5_changed_from_v7_8 ?? '?';

        const drift =
            d.outside_mask_drift ?? '?';

        const regression =
            d.non_print_v7_8_regression ?? '?';

        status(
            `A5-v1 complete. ` +
            `Eligible ${eligible} px; ` +
            `PRINT ${printPixels} px; ` +
            `effective change from V7.8 ${changed} px; ` +
            `outside-mask drift ${drift} px; ` +
            `non-PRINT V7.8 regression ${regression} px.`
        );

        $('whole').scrollIntoView({
            behavior: 'smooth'
        });

    } catch (error) {
        status(
            'Correction failed: ' +
            error.message
        );
    }
};


/* ============================================================
   WHOLE-PAGE REVIEW DECISION
   ============================================================ */

document
    .querySelectorAll('[data-decision]')
    .forEach(
        button => {
            button.onclick = async () => {
                const decision =
                    button.dataset.decision;

                /*
                 * A new review decision invalidates any download
                 * link from an earlier package.
                 */
                $('download').hidden = true;
                $('download').removeAttribute(
                    'href'
                );

                status(
                    `Saving whole-page decision: ${decision}...`
                );

                try {
                    const response =
                        await fetch(
                            `/api/session/${sid}/approval`,
                            {
                                method: 'POST',
                                headers: {
                                    'Content-Type':
                                        'application/json'
                                },
                                body: JSON.stringify({
                                    status: decision,
                                    note: $('note').value
                                })
                            }
                        );

                    const data =
                        await readResponse(
                            response
                        );

                    if (!response.ok) {
                        status(
                            'Whole-page decision failed: ' +
                            describeError(
                                data,
                                `HTTP ${response.status}`
                            )
                        );
                        return;
                    }

                    status(
                        `Whole-page decision saved: ${data.status}.`
                    );

                    if (
                        decision ===
                        'return_to_mask'
                    ) {
                        $('review').scrollIntoView({
                            behavior: 'smooth'
                        });
                    }

                } catch (error) {
                    status(
                        'Whole-page decision failed: ' +
                        error.message
                    );
                }
            };
        }
    );


/* ============================================================
   BUILD EVIDENCE PACKAGE
   ============================================================ */

$('package').onclick = async () => {
    /*
     * Never expose a stale download URL while a new evidence
     * package is being constructed.
     */
    $('download').hidden = true;
    $('download').removeAttribute(
        'href'
    );

    status(
        'Building evidence package...'
    );

    try {
        const response = await fetch(
            `/api/session/${sid}/finish`,
            {
                method: 'POST'
            }
        );

        const data = await readResponse(
            response
        );

        if (!response.ok) {
            status(
                'Evidence packaging failed: ' +
                describeError(
                    data,
                    `HTTP ${response.status}`
                )
            );
            return;
        }

        if (!data.download_url) {
            status(
                'Evidence packaging failed: ' +
                'server returned no download URL.'
            );
            return;
        }

        /*
         * The URL returned by FastAPI is authoritative.
         * Do not rebuild or modify it here.
         */
        $('download').href =
            data.download_url;

        $('download').hidden =
            false;

        const filename =
            data.package_filename ||
            'hf_04_5_training_and_audit.zip';

        const packageBytes =
            Number(
                data.package_bytes
            );

        const byteText =
            Number.isFinite(packageBytes)
                ? ` (${packageBytes.toLocaleString()} bytes)`
                : '';

        status(
            `Evidence package ready: ` +
            `${filename}${byteText}.`
        );

    } catch (error) {
        status(
            'Evidence packaging failed: ' +
            error.message
        );
    }
};


/* ============================================================
   DOWNLOAD EVIDENCE PACKAGE
   ============================================================ */

$('download').onclick = async event => {
    event.preventDefault();

    /*
     * Use the exact href supplied by /finish.
     */
    const url =
        $('download').getAttribute(
            'href'
        );

    if (!url) {
        status(
            'No evidence package is available to download.'
        );
        return;
    }

    status(
        'Downloading evidence package...'
    );

    try {
        const response =
            await fetch(url);

        if (!response.ok) {
            const data =
                await readResponse(
                    response
                );

            status(
                'Evidence download failed: ' +
                describeError(
                    data,
                    `HTTP ${response.status}`
                )
            );

            return;
        }

        /*
         * FileResponse normally returns application/zip.
         * Do not fail solely because a proxy/browser reports
         * application/octet-stream instead.
         */
        const contentType =
            response.headers.get(
                'content-type'
            ) || '';

        if (
            contentType &&
            !contentType.includes(
                'application/zip'
            ) &&
            !contentType.includes(
                'application/octet-stream'
            )
        ) {
            status(
                'Evidence download failed: ' +
                `unexpected content type ${contentType}.`
            );
            return;
        }

        const blob =
            await response.blob();

        if (
            blob.size === 0
        ) {
            status(
                'Evidence download failed: ' +
                'server returned an empty file.'
            );
            return;
        }

        const objectUrl =
            URL.createObjectURL(
                blob
            );

        const temporaryLink =
            document.createElement(
                'a'
            );

        temporaryLink.href =
            objectUrl;

        temporaryLink.download =
            'hf_04_5_training_and_audit.zip';

        temporaryLink.style.display =
            'none';

        document.body.appendChild(
            temporaryLink
        );

        temporaryLink.click();
        temporaryLink.remove();

        /*
         * Delay revocation long enough for the browser to start
         * consuming the object URL.
         */
        setTimeout(
            () => {
                URL.revokeObjectURL(
                    objectUrl
                );
            },
            1000
        );

        status(
            `Evidence package downloaded ` +
            `(${blob.size.toLocaleString()} bytes).`
        );

    } catch (error) {
        status(
            'Evidence download failed: ' +
            error.message
        );
    }
};


/* ============================================================
   INITIAL UI STATE
   ============================================================ */

setMode('pan');

$('download').hidden = true;
$('download').removeAttribute(
    'href'
);