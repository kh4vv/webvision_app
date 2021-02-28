import React, { Component } from "react";
import axios from "axios";

class Upload extends Component {
  state = {
    selectedFile: null,
    imgname: '',
    predic: '',
  };
  fileSelectedHandler = (event) => {
    this.setState({
      selectedFile: event.target.files[0],
    });
  };

  fileUploadHandler = () => {
    const fd = new FormData();
    fd.append("file", this.state.selectedFile);
    axios
      .post("http://localhost:9000/mnist_upload", fd, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      })
      .then((res) => {
        console.log(res);
      });
  };

  fileDownloadHandler = () => {
      axios
        .get("http://localhost:9000/prediction").then(response => {
            console.log(response)
            this.setState({imgname: response.data.filename, predic: response.data.pred})
            console.log(response.data.filename, response.data.pred)
        }).catch(error => {
            console.log(error)
        })
  };

  onClick = () => {
    this.fileUploadHandler();
    this.fileDownloadHandler();
  };

  render() {
    return (
      <div className="upload">
        <input type="file" onChange={this.fileSelectedHandler} />
        <button onClick={this.onClick}> Upload </button>
        <h3> Prediction: {this.state.predic} </h3>
        <h4> Filename: {this.state.imgname} </h4>
      </div>
    );
  }
}
export default Upload;
