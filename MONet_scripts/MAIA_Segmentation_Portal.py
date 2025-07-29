from __future__ import annotations

import importlib.resources
import json
import os
import sys
from pathlib import Path

import requests
import SimpleITK as sitk
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from MONet.auth import get_token, verify_valid_token_exists, welcome_message
from MONet.utils import get_available_models


class MAIAInferenceApp(QWidget):
    def __init__(self):
        super().__init__()
        # Get latest tag from GitHub for version
        try:
            resp = requests.get("https://api.github.com/repos/SimoneBendazzoli93/MONet-Bundle/tags", timeout=3)
            if resp.ok and resp.json():
                version = resp.json()[0]["name"]
            else:
                version = "unknown"
        except Exception:
            version = "unknown"
        self.setWindowTitle("MAIA Segmentation Portal - {}".format(version))
        self.resize(400, 200)
        self.token = None
        self.models = {}
        try:
            os.makedirs(os.path.expanduser("~/.monet"), exist_ok=True)
            # Check if an auth file exists for the user
            auth_files = [f for f in os.listdir(os.path.expanduser("~/.monet")) if f.endswith("_auth.json")]
            if auth_files:
                print("Found auth files:", auth_files)
                self.username = auth_files[0].replace("_auth.json", "")
        except Exception:
            self.username = ""

        self.username_input = QLineEdit()
        self.username_input.setText(self.username)
        self.password_input = QLineEdit()
        self.login_button = QPushButton("Login")

        self.input_path_input = QLineEdit()
        self.output_path_input = QLineEdit()
        self.model_dropdown = QComboBox()
        self.infer_button = QPushButton("Run Inference")
        btn_font = self.infer_button.font()
        btn_font.setPointSize(14)
        btn_font.setFamily("Ubuntu")
        self.infer_button.setFont(btn_font)

        self.init_login_ui()

    def init_login_ui(self):
        layout = QVBoxLayout()

        layout.addWidget(QLabel("Username:"))
        layout.addWidget(self.username_input)

        layout.addWidget(QLabel("Password:"))
        self.password_input.setEchoMode(QLineEdit.Password)
        layout.addWidget(self.password_input)

        self.login_button.clicked.connect(self.login)
        layout.addWidget(self.login_button)

        self.setLayout(layout)

    def login(self):
        username = self.username_input.text().strip()
        password = self.password_input.text().strip()

        if not username:
            QMessageBox.warning(self, "Input Error", "Username is required.")
            return

        auth_path = os.path.expanduser(f"~/.monet/{username}_auth.json")

        try:
            if not verify_valid_token_exists(username):
                if not password:
                    QMessageBox.warning(self, "Input Error", "Password required for first login.")
                    return
                token_data = get_token(username, password)
                with open(auth_path, "w") as f:
                    json.dump(token_data, f)
                self.token = token_data["access_token"]
                QMessageBox.information(self, "Welcome", welcome_message(self.token))
            else:
                with open(auth_path) as f:
                    token_data = json.load(f)
                    self.token = token_data["access_token"]
        except Exception as e:
            QMessageBox.critical(self, "Login Failed", str(e))
            return

        self.username = username
        self.models = get_available_models(self.token, username)
        self.model_dropdown.addItems(list(self.models.keys()))
        self.init_main_ui()

    def init_main_ui(self):
        self.setWindowTitle("MAIA Segmentation Portal - Home")

        # Remove all widgets and layouts from the current layout (handle nested layouts)
        def clear_layout(layout):
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                child_layout = item.layout()
                if widget is not None:
                    widget.setParent(None)
                elif child_layout is not None:
                    clear_layout(child_layout)

        clear_layout(self.layout())
        layout = self.layout()

        # Add a logo at the top
        logo_label = QLabel()
        logo_label.setAlignment(Qt.AlignCenter)

        with importlib.resources.path("MONet.icons", "logo.svg") as icon_path:
            logo_label.setPixmap(QIcon(str(icon_path)).pixmap(120, 120))

        logo_label.setCursor(Qt.PointingHandCursor)

        def open_maia_website(event):
            url = "https://maia.app.cloud.cbh.kth.se"
            if sys.platform.startswith("win"):
                os.startfile(url)
            elif sys.platform.startswith("darwin"):
                os.system(f'open "{url}"')
            else:
                os.system(f'xdg-open "{url}"')

        logo_label.mousePressEvent = open_maia_website
        layout.addWidget(logo_label)
        welcome_label = QLabel(f"Welcome to MAIA Segmentation Portal, {self.username}! ")
        welcome_label_2 = QLabel("Select an option below: ")
        font = welcome_label.font()
        font.setPointSize(16)
        font.setFamily("Ubuntu")  # Example of a fancy font, you can choose another
        welcome_label.setFont(font)
        welcome_label.setAlignment(Qt.AlignCenter)
        welcome_label_2.setFont(font)
        welcome_label_2.setAlignment(Qt.AlignCenter)
        layout.addWidget(welcome_label)
        # Add a separator line between the welcome texts
        separator = QLabel()
        separator.setFrameShape(QLabel.HLine)
        separator.setFrameShadow(QLabel.Sunken)
        layout.addWidget(separator)
        layout.addWidget(welcome_label_2)

        remote_infer_btn = QPushButton("Remote Inference")
        remote_infer_btn.clicked.connect(self.init_inference_ui)
        remote_infer_btn.setSizePolicy(
            remote_infer_btn.sizePolicy().horizontalPolicy(), remote_infer_btn.sizePolicy().verticalPolicy()
        )

        remote_infer_btn.setMinimumWidth(remote_infer_btn.sizeHint().width() * 2)
        remote_infer_btn.adjustSize()
        # Set custom font for button text
        btn_font = remote_infer_btn.font()
        btn_font.setPointSize(14)
        btn_font.setFamily("Ubuntu")
        remote_infer_btn.setFont(btn_font)

        local_infer_btn = QPushButton("Local Inference")
        local_infer_btn.clicked.connect(self.local_inference_ui)
        local_infer_btn.setSizePolicy(
            local_infer_btn.sizePolicy().horizontalPolicy(), local_infer_btn.sizePolicy().verticalPolicy()
        )
        local_infer_btn.setFont(btn_font)
        local_infer_btn.setMinimumHeight(local_infer_btn.sizeHint().height() * 2)
        local_infer_btn.setMinimumWidth(local_infer_btn.sizeHint().width() * 2)
        local_infer_btn.adjustSize()

        model_info_btn = QPushButton("Available Models")
        model_info_btn.clicked.connect(self.init_models_info)
        model_info_btn.setSizePolicy(model_info_btn.sizePolicy().horizontalPolicy(), model_info_btn.sizePolicy().verticalPolicy())

        model_info_btn.adjustSize()
        model_info_btn.setFont(btn_font)

        concat_modalities_btn = QPushButton("Concatenate Modalities")
        concat_modalities_btn.clicked.connect(self.init_concat_ui)
        concat_modalities_btn.setSizePolicy(
            concat_modalities_btn.sizePolicy().horizontalPolicy(), concat_modalities_btn.sizePolicy().verticalPolicy()
        )
        concat_modalities_btn.setMinimumHeight(concat_modalities_btn.sizeHint().height() * 2)
        concat_modalities_btn.setMinimumWidth(concat_modalities_btn.sizeHint().width() * 2)
        concat_modalities_btn.adjustSize()
        concat_modalities_btn.setFont(btn_font)

        
        button_height = remote_infer_btn.sizeHint().height() * 2
        button_width = remote_infer_btn.sizeHint().width() * 2

        remote_infer_btn.setMinimumHeight(button_height)
        remote_infer_btn.setMinimumWidth(button_width)
        remote_infer_btn.adjustSize()

        local_infer_btn.setMinimumHeight(button_height)
        local_infer_btn.setMinimumWidth(button_width)
        local_infer_btn.adjustSize()

        model_info_btn.setMinimumHeight(button_height)
        model_info_btn.setMinimumWidth(button_width)
        model_info_btn.adjustSize()

        concat_modalities_btn.setMinimumHeight(button_height)
        concat_modalities_btn.setMinimumWidth(button_width)
        concat_modalities_btn.adjustSize()
        
        # Add icon to the Remote Inference button
        with importlib.resources.path("MONet.icons", "Remote.png") as icon_path:
            remote_infer_btn.setIcon(QIcon(str(icon_path)))
            remote_infer_btn.setIconSize(remote_infer_btn.size())
            remote_infer_btn.setStyleSheet("text-align: left; padding-left: 40px;")  # Align icon left, add padding for text
        layout.addWidget(remote_infer_btn)
        # Add icon to the Local Inference button
        with importlib.resources.path("MONet.icons", "Local.png") as icon_path:
            local_infer_btn.setIcon(QIcon(str(icon_path)))
            local_infer_btn.setIconSize(local_infer_btn.size())
            local_infer_btn.setStyleSheet("text-align: left; padding-left: 40px;")  # Align icon left, add padding for text
        layout.addWidget(local_infer_btn)
        
        with importlib.resources.path("MONet.icons", "Models.png") as icon_path:
            model_info_btn.setIcon(QIcon(str(icon_path)))
            model_info_btn.setIconSize(model_info_btn.size())
            model_info_btn.setStyleSheet("text-align: left; padding-left: 40px;")  # Align icon left, add padding for text
        layout.addWidget(model_info_btn)
        with importlib.resources.path("MONet.icons", "Concatenate.png") as icon_path:
            concat_modalities_btn.setIcon(QIcon(str(icon_path)))
            concat_modalities_btn.setIconSize(concat_modalities_btn.size())
            concat_modalities_btn.setStyleSheet("text-align: left; padding-left: 40px;")  # Align icon left, add padding for text
        layout.addWidget(concat_modalities_btn)

    def init_inference_ui(self):
        self.setWindowTitle("MAIA Segmentation Portal - Remote Inference")

        def clear_layout(layout):
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                child_layout = item.layout()
                if widget is not None:
                    widget.setParent(None)
                elif child_layout is not None:
                    clear_layout(child_layout)

        clear_layout(self.layout())

        layout = self.layout()  # QVBoxLayout()

        home_button = QPushButton("")
        home_button.setFixedSize(40, 40)
        with importlib.resources.path("MONet.icons", "Home-icon.svg.png") as icon_path:
            home_button.setIcon(QIcon(str(icon_path)))  # or .png
        home_button.setIconSize(home_button.size())
        home_button.setToolTip("Home")
        # Alternatively, use a local icon file:
        # home_button.setIcon(QIcon("/path/to/home_icon.png"))
        home_button.clicked.connect(self.init_main_ui)
        layout.addWidget(home_button)
        label_input = QLabel("1. Select Input File:")
        font = label_input.font()
        font.setPointSize(14)
        font.setFamily("Ubuntu")
        label_input.setFont(font)
        layout.addWidget(label_input)
        layout.addWidget(self.input_path_input)
        browse_input = QPushButton("Browse")
        browse_input.clicked.connect(self.browse_input)
        layout.addWidget(browse_input)
        separator = QLabel()
        separator.setFrameShape(QLabel.HLine)
        separator.setFrameShadow(QLabel.Sunken)
        layout.addWidget(separator)
        label_output = QLabel("2. Select Output File:")
        font_output = label_output.font()
        font_output.setPointSize(14)
        font_output.setFamily("Ubuntu")
        label_output.setFont(font_output)
        layout.addWidget(label_output)
        layout.addWidget(self.output_path_input)
        browse_output = QPushButton("Browse")
        browse_output.clicked.connect(self.browse_output)
        layout.addWidget(browse_output)
        separator2 = QLabel()
        separator2.setFrameShape(QLabel.HLine)
        separator2.setFrameShadow(QLabel.Sunken)
        layout.addWidget(separator2)
        label_model = QLabel("3. Choose Model:")
        font_model = label_model.font()
        font_model.setPointSize(14)
        font_model.setFamily("Ubuntu")
        label_model.setFont(font_model)
        layout.addWidget(label_model)
        if len(self.models.keys()) == 0:
            self.models = get_available_models(self.token, self.username)
            self.model_dropdown.addItems(list(self.models.keys()))
        layout.addWidget(self.model_dropdown)

        self.infer_button.clicked.connect(self.run_inference)
        layout.addWidget(self.infer_button)

        self.setLayout(layout)

    def local_inference_ui(self):
        self.setWindowTitle("MAIA Segmentation Portal - Local Inference")

        def clear_layout(layout):
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                child_layout = item.layout()
                if widget is not None:
                    widget.setParent(None)
                elif child_layout is not None:
                    clear_layout(child_layout)

        clear_layout(self.layout())

        layout = self.layout()  # QVBoxLayout()

        home_button = QPushButton("")
        home_button.setFixedSize(40, 40)
        with importlib.resources.path("MONet.icons", "Home-icon.svg.png") as icon_path:
            home_button.setIcon(QIcon(str(icon_path)))  # or .png
        home_button.setIconSize(home_button.size())
        home_button.setToolTip("Home")
        # Alternatively, use a local icon file:
        # home_button.setIcon(QIcon("/path/to/home_icon.png"))
        home_button.clicked.connect(self.init_main_ui)
        layout.addWidget(home_button)
        label_input = QLabel("1. Select Input File:")
        font = label_input.font()
        font.setPointSize(14)
        font.setFamily("Ubuntu")
        label_input.setFont(font)
        layout.addWidget(label_input)
        layout.addWidget(self.input_path_input)
        browse_input = QPushButton("Browse")
        browse_input.clicked.connect(self.browse_input)
        layout.addWidget(browse_input)
        separator = QLabel()
        separator.setFrameShape(QLabel.HLine)
        separator.setFrameShadow(QLabel.Sunken)
        layout.addWidget(separator)
        label_output = QLabel("2. Select Output Folder:")
        font_output = label_output.font()
        font_output.setPointSize(14)
        font_output.setFamily("Ubuntu")
        label_output.setFont(font_output)
        layout.addWidget(label_output)
        layout.addWidget(self.output_path_input)
        browse_output = QPushButton("Browse")
        browse_output.clicked.connect(self.browse_output_folder)
        layout.addWidget(browse_output)
        separator2 = QLabel()
        separator2.setFrameShape(QLabel.HLine)
        separator2.setFrameShadow(QLabel.Sunken)
        layout.addWidget(separator2)
        label_model = QLabel("3. Choose Model:")
        font_model = label_model.font()
        font_model.setPointSize(14)
        font_model.setFamily("Ubuntu")
        label_model.setFont(font_model)
        layout.addWidget(label_model)
        if len(self.models.keys()) == 0:
            self.models = get_available_models(self.token, self.username)
            self.model_dropdown.addItems(list(self.models.keys()))
        layout.addWidget(self.model_dropdown)

        self.infer_button.clicked.connect(self.run_local_inference)
        layout.addWidget(self.infer_button)

        self.setLayout(layout)

    def run_local_inference(self):
        try:
            from MONet_scripts.MONet_local_inference import run_inference as local_inference_main

            output_image = local_inference_main(
                self.model_dropdown.currentText(), self.username, self.input_path_input.text(), self.output_path_input.text()
            )
            QMessageBox.information(
                self, "Inference Completed", f"Segmentation saved to {Path(self.output_path_input.text()).joinpath(output_image)}"
            )
        except ImportError:
            QMessageBox.critical(
                self, "Import Error", "Failed to import the local inference function. Please ensure MONet package is installed."
            )
            return

    def init_models_info(self):
        if len(self.models.keys()) == 0:
            self.models = get_available_models(self.token, self.username)
        # Gather info for all models
        model_infos = []
        for model in self.models:
            response = requests.get(f"{self.models[model]}info/", headers={"Authorization": f"Bearer {self.token}"})
            if response.status_code == 200:
                model_info = response.json()["models"]["MONetBundle"]
                labels = ", ".join(model_info.get("labels", []))
                description = model_info.get("description", "")
                inputs = ", ".join(model_info.get("metadata", {}).get("inputs", []))
                model_infos.append((model, labels, description, inputs))
            else:
                print(f"Failed to retrieve info for model {model}")

        if model_infos:
            dialog = QDialog(self)
            dialog.setWindowTitle("Available Models")
            table = QTableWidget(len(model_infos), 4)
            table.setHorizontalHeaderLabels(["Model", "Description", "Inputs", "Labels"])
            for row, (model, labels, description, inputs) in enumerate(model_infos):
                table.setItem(row, 0, QTableWidgetItem(model))
                table.setItem(row, 1, QTableWidgetItem(description))
                table.setItem(row, 2, QTableWidgetItem(inputs))
                table.setItem(row, 3, QTableWidgetItem(labels))
                table.resizeColumnsToContents()

            layout = QVBoxLayout()
            layout.addWidget(table)
            dialog.setLayout(layout)
            table.resizeColumnsToContents()
            table.resizeRowsToContents()
            dialog.resize(table.horizontalHeader().length() + 40, table.verticalHeader().length() + 80)
            dialog.exec_()
        else:
            print(f"Failed to retrieve info for model {model}")

    def init_concat_ui(self):
        self.setWindowTitle("MAIA Segmentation Portal - Concatenate Modalities")

        def clear_layout(layout):
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                child_layout = item.layout()
                if widget is not None:
                    widget.setParent(None)
                elif child_layout is not None:
                    clear_layout(child_layout)

        clear_layout(self.layout())
        layout = self.layout()

        home_button = QPushButton("")
        home_button.setFixedSize(40, 40)
        with importlib.resources.path("MONet.icons", "Home-icon.svg.png") as icon_path:
            home_button.setIcon(QIcon(str(icon_path)))  # or .png
        home_button.setIconSize(home_button.size())
        home_button.setToolTip("Home")
        home_button.clicked.connect(self.init_main_ui)
        layout.addWidget(home_button)

        if len(self.models.keys()) == 0:
            self.models = get_available_models(self.token, self.username)
            self.model_dropdown.addItems(list(self.models.keys()))
        layout.addWidget(self.model_dropdown)

        # Container for input selectors
        self.input_selectors = []

        # Widget to hold input selectors
        self.inputs_widget = QWidget()
        self.inputs_layout = QVBoxLayout()
        self.inputs_widget.setLayout(self.inputs_layout)
        layout.addWidget(self.inputs_widget)

        def update_input_selectors():
            # Clear previous selectors
            for selector in self.input_selectors:
                self.inputs_layout.removeWidget(selector)
                selector.deleteLater()
            self.input_selectors = []

            model = self.model_dropdown.currentText()
            if not model:
                return

            # Fetch model info
            try:
                response = requests.get(f"{self.models[model]}info/", headers={"Authorization": f"Bearer {self.token}"})
                response.raise_for_status()
                model_metadata = response.json()["models"]["MONetBundle"]["metadata"]
                required_channels = model_metadata.get("inputs", [])
            except Exception as e:
                QMessageBox.critical(self, "Model Info Error", f"Could not fetch model info: {e}")
                return

            # Create selectors for each required input
            for idx, channel in enumerate(required_channels):
                selector_layout = QHBoxLayout()
                label = QLabel(f"Input {idx+1} ({channel}):")
                line_edit = QLineEdit()
                browse_btn = QPushButton("Browse")

                def browse_file(le):
                    path, _ = QFileDialog.getOpenFileName(self, "Select Input File", "", "NIfTI Files (*.nii.gz)")
                    if path:
                        le.setText(path)

                browse_btn.clicked.connect(lambda _, le=line_edit: browse_file(le))
                selector_layout.addWidget(label)
                selector_layout.addWidget(line_edit)
                selector_layout.addWidget(browse_btn)
                container = QWidget()
                container.setLayout(selector_layout)
                # Store both the container and the line_edit for easy access
                container.line_edit = line_edit
                self.inputs_layout.addWidget(container)
                self.input_selectors.append(container)

        self.model_dropdown.currentIndexChanged.connect(update_input_selectors)
        update_input_selectors()

        # Reference modality selector
        ref_modality_layout = QHBoxLayout()
        ref_modality_label = QLabel("Reference Modality:")
        self.ref_modality_dropdown = QComboBox()
        # Populate with required channels/modalities for the selected model
        model = self.model_dropdown.currentText()
        if model:
            try:
                response = requests.get(f"{self.models[model]}info/", headers={"Authorization": f"Bearer {self.token}"})
                response.raise_for_status()
                model_metadata = response.json()["models"]["MONetBundle"]["metadata"]
                required_channels = model_metadata.get("inputs", [])
                self.ref_modality_dropdown.addItems(required_channels)
            except Exception:
                self.ref_modality_dropdown.addItem("Unknown")
        ref_modality_layout.addWidget(ref_modality_label)
        ref_modality_layout.addWidget(self.ref_modality_dropdown)
        layout.addLayout(ref_modality_layout)

        # Update reference modality dropdown when model changes
        def update_ref_modality_dropdown():
            self.ref_modality_dropdown.clear()
            model = self.model_dropdown.currentText()
            if model:
                try:
                    response = requests.get(f"{self.models[model]}info/", headers={"Authorization": f"Bearer {self.token}"})
                    response.raise_for_status()
                    model_metadata = response.json()["models"]["MONetBundle"]["metadata"]
                    required_channels = model_metadata.get("inputs", [])
                    self.ref_modality_dropdown.addItems(required_channels)
                except Exception:
                    self.ref_modality_dropdown.addItem("Unknown")

        self.model_dropdown.currentIndexChanged.connect(update_ref_modality_dropdown)

        # Output file selector
        output_layout = QHBoxLayout()
        output_label = QLabel("Output File:")
        self.output_path_input = QLineEdit()
        output_browse_btn = QPushButton("Browse")

        def browse_output_file():
            path = QFileDialog.getExistingDirectory(self, "Select Output Folder", "")
            if path:
                self.output_path_input.setText(path)

        output_browse_btn.clicked.connect(browse_output_file)
        output_layout.addWidget(output_label)
        output_layout.addWidget(self.output_path_input)
        output_layout.addWidget(output_browse_btn)
        layout.addLayout(output_layout)

        # Concatenate button
        concat_button = QPushButton("Concatenate Modalities")
        concat_button.setFont(self.infer_button.font())
        concat_button.clicked.connect(self.concatenate_modalities)
        layout.addWidget(concat_button)

    def concatenate_modalities(self):
        input_files = []
        for selector in self.input_selectors:
            # Use the stored line_edit attribute
            line_edit = getattr(selector, "line_edit", None)
            if line_edit is not None:
                input_files.append(line_edit.text())
            else:
                input_files.append("")

        output_folder = self.output_path_input.text()

        reference_modality = self.ref_modality_dropdown.currentText()

        modalities = []
        for selector in self.input_selectors:
            line_edit = getattr(selector, "line_edit", None)
            if line_edit is not None:
                input_file = line_edit.text()
                # Get the channel name from the label in the selector layout
                label = selector.layout().itemAt(0).widget()
                modality = label.text().split("(")[-1].rstrip("):")
                modalities.append(modality)
                print(f"Input file: {input_file}, Modality: {modality}")
            # Now you have both input_file and modality for each input
        try:
            from MONet_scripts.MONet_concatenate_modalities import concatenate

            data = {modality: input_file for modality, input_file in zip(modalities, input_files)}
            print(
                f"Concatenating modalities: {data} with reference modality: {reference_modality} to output folder: {output_folder}"
            )
            QMessageBox.information(self, "Concatenation Started", "Concatenating modalities, please wait...")
            output_file = concatenate(data, reference_modality, output_folder)
            QMessageBox.information(
                self, "Success", f"Modalities concatenated and saved to {Path(output_folder).joinpath(output_file)}"
            )
        except ImportError:
            QMessageBox.critical(
                self, "Import Error", "Failed to import the concatenation function. Please ensure MONet package is installed."
            )
            return

    def browse_input(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Input File", "", "NIfTI Files (*.nii.gz)")
        if path:
            self.input_path_input.setText(path)

    def browse_output(self):
        path, _ = QFileDialog.getSaveFileName(self, "Select Output File", "", "NIfTI Files (*.nii.gz)")
        if path:
            self.output_path_input.setText(path)

    def browse_output_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Folder", "")
        if path:
            self.output_path_input.setText(path)

    def run_inference(self):
        input_file = self.input_path_input.text()
        output_file = self.output_path_input.text()
        model = self.model_dropdown.currentText()

        if not all([input_file, output_file, model]):
            QMessageBox.warning(self, "Missing Fields", "Please complete all fields.")
            return

        base_url = self.models[model]
        info_url = f"{base_url}info/"
        infer_url = f"{base_url}infer/MONetBundle?output=image"
        headers = {"Authorization": f"Bearer {self.token}"}

        try:
            response = requests.get(info_url, headers=headers)
            response.raise_for_status()
            model_metadata = response.json()["models"]["MONetBundle"]["metadata"]
            required_channels = model_metadata["inputs"]

            img = sitk.ReadImage(input_file)
            num_channels = 1 if len(img.GetSize()) < 4 else img.GetSize()[3]

            if num_channels != len(required_channels):
                QMessageBox.critical(
                    self,
                    "Input Error",
                    f"The model you selected ({model}) expected {len(required_channels)} channels, got {num_channels}.",
                )
                return

            with open(input_file, "rb") as f:
                files = {
                    "params": (None, json.dumps({}, indent=2), "application/json"),
                    "file": (os.path.basename(input_file), f, "application/gzip"),
                }
                res = requests.post(infer_url, headers=headers, files=files)
                res.raise_for_status()

                with open(output_file, "wb") as out:
                    out.write(res.content)

                QMessageBox.information(self, "Success", f"Output saved to {output_file}")
        except Exception as e:
            QMessageBox.critical(self, "Inference Failed", str(e))


def main():

    app = QApplication(sys.argv)
    window = MAIAInferenceApp()
    with importlib.resources.path("MONet.icons", "logo.svg") as icon_path:
        app.setWindowIcon(QIcon(str(icon_path)))
        window.setWindowIcon(QIcon(str(icon_path)))
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
