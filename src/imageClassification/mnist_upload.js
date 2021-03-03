import React, { Component } from "react";
import axios from "axios";

class Upload extends Component {
  state = {
    selectedFile: null,
    imgname: '',
    predic: '',
    imagePreviewUrl: ''
  };

  fileSelectedHandler = (event) => {
    event.preventDefault();
    let reader = new FileReader();
    this.setState({
      selectedFile: event.target.files[0],
    });

    reader.onload = () => {
      this.setState({
        imagePreviewUrl: reader.result
      });
    };

    reader.readAsDataURL(event.target.files[0]);
  };

  fileUploadHandler = async event => {
    event.preventDefault();
    const { imgname } = this.state;
    const { predic } = this.state;
    const { setData } = this.props;

    if (!this.state.selectedFile) {
      alert("Please Upload Image.")
      return;
    }

    const fd = new FormData();
    fd.append("file", this.state.selectedFile);
    const alldata = await axios
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
      .get("http://localhost:9000/mnist_upload").then(response => {
        console.log(response)
        this.setState({ imgname: response.data.filename, predic: response.data.pred })
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

    let { imagePreviewUrl } = this.state;
    let $imagePreview = null;
    if (imagePreviewUrl) {
      $imagePreview = <img src={imagePreviewUrl} alt="" />;
      console.log(imagePreviewUrl)
    }


    return (
      <div className="upload">
        <input type="file" onChange={this.fileSelectedHandler} />
        <button type='submit' onClick={e => this.fileUploadHandler(e)}> Upload </button>
        <div className="imgPreview">{$imagePreview}</div>
        <h3> Prediction: {this.state.predic} </h3>
        <h4> Filename: {this.state.imgname} </h4>
      </div>
    );
  }
}
export default Upload;
