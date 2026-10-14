import json
import gradio as gr


class ReviewController:

    def __init__(
        self,
        queue_manager,
        loader,
        saver
    ):

        self.queue_manager = (
            queue_manager
        )

        self.loader = loader

        self.saver = saver

        self.refresh()

    def refresh(self):

        self.queue = (
            self.queue_manager
            .build_queue()
        )

        self.queue_names = [
            p.name
            for p in self.queue
        ]

        self.queue_map = {
            p.name: p
            for p in self.queue
        }

    def load(self, name):

        return json.dumps(
            self.loader.package_page(
                name
            )
        )

    def skip(self, name):

        if len(
            self.queue_names
        ) > 1:

            self.queue_names.remove(
                name
            )

            self.queue_names.append(
                name
            )

        return self.load(
            self.queue_names[0]
        )
