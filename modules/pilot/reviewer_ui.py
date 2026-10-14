# modules/pilot/reviewer_ui.py

import gradio as gr


class ReviewerUI:

    CSS = """
    .gradio-container{
        max-width:100vw!important;
        padding:6px!important;
    }

    .compact img{
        max-height:58vh!important;
        object-fit:contain!important;
    }

    .compact-table{
        max-height:28vh!important;
        overflow:auto!important;
    }

    footer{
        display:none!important;
    }
    """

    def __init__(
        self,
        backend,
        final_dir,
        public_link=True
    ):

        self.backend = backend

        self.final_dir = final_dir

        self.public_link = public_link

        self.pages = sorted(
            backend.state.keys()
        )

    # --------------------------------------------------
    # Refresh
    # --------------------------------------------------

    def refresh(
        self,
        page,
        category,
        status,
        view
    ):

        overview = self.backend.overview(
            page,
            category,
            status,
            view,
        )

        table = self.backend.table(
            page
        )

        choices = self.backend.queue_choices(
            page
        )

        selected = (
            self.backend
            .first_unreviewed(page)
        )

        counts = self.backend.counts(
            page
        )

        summary = (
            f"Total: {len(choices)} | "
            f"Unreviewed: {counts['unreviewed']} | "
            f"Deferred: {counts['deferred']} | "
            f"Accepted: {counts['accepted']} | "
            f"Unchanged: {counts['left_unchanged']}"
        )

        return (

            overview,

            table,

            gr.update(
                choices=choices,
                value=selected
            ),

            summary

        )

    # --------------------------------------------------
    # Component
    # --------------------------------------------------

    def component_details(
        self,
        page,
        component
    ):

        return (
            self.backend
            .component_diagnostic(
                page,
                component
            )
        )

    # --------------------------------------------------
    # Decisions
    # --------------------------------------------------

    def make_decision(
        self,
        page,
        component,
        notes,
        decision
    ):

        return self.backend.decide(
            page,
            component,
            decision,
            notes,
        )

    # --------------------------------------------------
    # Approve page
    # --------------------------------------------------

    def approve(
        self,
        page,
        notes
    ):

        output = (

            self.final_dir
            /
            f"{page.split('.')[0]}.png"

        )

        image = self.backend.approve(
            page,
            output,
            notes,
        )

        return (

            image,

            f"Approved: {output}"

        )

    # --------------------------------------------------
    # Build interface
    # --------------------------------------------------

    def build(self):

        first_page = (
            self.pages[0]
            if self.pages
            else None
        )

        with gr.Blocks(
            title="Pilot Reviewer",
            css=self.CSS
        ) as app:

            gr.Markdown(
                "# Pilot Reviewer"
            )

            with gr.Row():

                page = gr.Dropdown(
                    self.pages,
                    value=first_page,
                    label="Page",
                )

                category = gr.Radio(
                    [
                        "all",
                        "green",
                        "amber",
                        "red"
                    ],
                    value="all",
                    label="Category",
                )

                status = gr.Radio(
                    [
                        "all",
                        "unreviewed",
                        "accepted",
                        "deferred",
                        "left_unchanged"
                    ],
                    value="all",
                    label="Status",
                )

                view = gr.Radio(
                    [
                        "current",
                        "proposal",
                        "original"
                    ],
                    value="current",
                    label="View",
                )

            summary = gr.Textbox(
                label="Queue Status",
                interactive=False,
            )

            with gr.Tab("Overview"):

                overview = gr.Image(
                    label="Overview"
                )

                queue_table = (
                    gr.Dataframe(
                        interactive=False,
                        label="Components"
                    )
                )

            with gr.Tab(
                "Component Review"
            ):

                component = gr.Dropdown(
                    label="Component"
                )

                diagnostic = gr.Image(
                    label="Diagnostic"
                )

                diagnostic_text = (
                    gr.Textbox(
                        label="Detail",
                        interactive=False,
                    )
                )

                notes = gr.Textbox(
                    label="Notes"
                )

                decision_status = (
                    gr.Textbox(
                        interactive=False,
                        label="Action"
                    )
                )

                with gr.Row():

                    accept = gr.Button(
                        "Accept",
                        variant="primary"
                    )

                    unchanged = gr.Button(
                        "Leave Unchanged"
                    )

                    defer = gr.Button(
                        "Defer"
                    )

            with gr.Tab(
                "Final Approval"
            ):

                approval_notes = (
                    gr.Textbox(
                        label="Approval Notes"
                    )
                )

                approve_btn = (
                    gr.Button(
                        "Approve Page",
                        variant="primary"
                    )
                )

                approved_image = (
                    gr.Image(
                        label="Final Output"
                    )
                )

                approval_status = (
                    gr.Textbox(
                        interactive=False
                    )
                )

            # ----------------------------------
            # Refresh
            # ----------------------------------

            refresh_outputs = [

                overview,

                queue_table,

                component,

                summary,
            ]

            for control in [

                page,
                category,
                status,
                view,

            ]:

                control.change(

                    self.refresh,

                    inputs=[
                        page,
                        category,
                        status,
                        view,
                    ],

                    outputs=refresh_outputs,
                )

            app.load(

                self.refresh,

                inputs=[
                    page,
                    category,
                    status,
                    view,
                ],

                outputs=refresh_outputs,
            )

            # ----------------------------------
            # Component diagnostics
            # ----------------------------------

            component.change(

                self.component_details,

                inputs=[
                    page,
                    component,
                ],

                outputs=[
                    diagnostic,
                    diagnostic_text,
                ],
            )

            # ----------------------------------
            # Decisions
            # ----------------------------------

            accept.click(

                lambda p, c, n:
                self.make_decision(
                    p,
                    c,
                    n,
                    "accepted"
                ),

                inputs=[
                    page,
                    component,
                    notes,
                ],

                outputs=
                decision_status,
            )

            unchanged.click(

                lambda p, c, n:
                self.make_decision(
                    p,
                    c,
                    n,
                    "left_unchanged"
                ),

                inputs=[
                    page,
                    component,
                    notes,
                ],

                outputs=
                decision_status,
            )

            defer.click(

                lambda p, c, n:
                self.make_decision(
                    p,
                    c,
                    n,
                    "deferred"
                ),

                inputs=[
                    page,
                    component,
                    notes,
                ],

                outputs=
                decision_status,
            )

            # ----------------------------------
            # Approval
            # ----------------------------------

            approve_btn.click(

                self.approve,

                inputs=[
                    page,
                    approval_notes,
                ],

                outputs=[
                    approved_image,
                    approval_status,
                ],
            )

        return app

    # --------------------------------------------------
    # Launch
    # --------------------------------------------------

    def launch(self):

        app = self.build()

        app.launch(
            share=self.public_link,
            debug=False,
            show_error=True,
        )
