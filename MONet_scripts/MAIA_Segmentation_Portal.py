import os
import json
import requests
import SimpleITK as sitk
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog, QComboBox, QMessageBox
)
from PyQt5.QtCore import Qt
import sys
from PyQt5.QtGui import QIcon
from MONet.auth import get_token, verify_valid_token_exists, welcome_message
from MONet.utils import get_available_models
import importlib.resources
class MAIAInferenceApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MAIA Segmentation Portal")
        self.resize(400, 200)
        self.token = None
        self.models = {}
        self.username = ""

        self.username_input = QLineEdit()
        self.password_input = QLineEdit()
        self.login_button = QPushButton("Login")

        self.input_path_input = QLineEdit()
        self.output_path_input = QLineEdit()
        self.model_dropdown = QComboBox()
        self.infer_button = QPushButton("Run Remote Inference")
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
        
        for i in reversed(range(self.layout().count())):
            self.layout().itemAt(i).widget().setParent(None)
        layout = self.layout()

        # Add a logo at the top
        logo_label = QLabel()
        logo_label.setAlignment(Qt.AlignCenter)
        
        with importlib.resources.path("MONet.icons", "logo.svg") as icon_path:
            logo_label.setPixmap(QIcon(icon_path).pixmap(120, 120))
            
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
        remote_infer_btn.setSizePolicy(remote_infer_btn.sizePolicy().horizontalPolicy(), remote_infer_btn.sizePolicy().verticalPolicy())
        remote_infer_btn.adjustSize()
        # Set custom font for button text
        btn_font = remote_infer_btn.font()
        btn_font.setPointSize(14)
        btn_font.setFamily("Ubuntu")
        remote_infer_btn.setFont(btn_font)
        
        layout.addWidget(remote_infer_btn)

    def init_inference_ui(self):
        self.setWindowTitle("MAIA Segmentation Portal - Remote Inference")
        for i in reversed(range(self.layout().count())):
            self.layout().itemAt(i).widget().setParent(None)

        layout = self.layout() #QVBoxLayout()

        
        home_button = QPushButton("")
        home_button.setFixedSize(40, 40)
        with importlib.resources.path("MONet.icons", "Home-icon.svg.png") as icon_path:
            home_button.setIcon(QIcon(icon_path))  # or .png
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

    def browse_input(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Input File", "", "NIfTI Files (*.nii.gz)")
        if path:
            self.input_path_input.setText(path)

    def browse_output(self):
        path, _ = QFileDialog.getSaveFileName(self, "Select Output File", "", "NIfTI Files (*.nii.gz)")
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
                QMessageBox.critical(self, "Input Error", f"Expected {len(required_channels)} channels, got {num_channels}.")
                return

            with open(input_file, "rb") as f:
                files = {
                    "params": (None, json.dumps({}, indent=2), "application/json"),
                    "file": (os.path.basename(input_file), f, "application/gzip")
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
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()