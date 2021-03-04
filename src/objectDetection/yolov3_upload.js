import React, { Component } from "react";
import axios from "axios";
import { Icon } from "semantic-ui-react";

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

    if (!this.state.selectedFile) {
      alert("Please Upload Image.")
      return;
    }

    const fd = new FormData();
    fd.append("file", this.state.selectedFile);
    await axios
      .post("http://localhost:9000/yolov3", fd, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      })
      .then((res) => {
        console.log(res);
        this.setState({ imgname: res.data.filename});
        console.log(this.state.imgname);
      });
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
        <h2><Icon name="file image" />Filename: {this.state.imgname} </h2>
        {this.state.imgname
          ? <img className={this.state.imgname}
            src={'http://localhost:9000/outputs/' + this.state.imgname} alt="" style={{ width: '100%' }} />
          : null}
      </div>
    );
  }
}
export default Upload;
