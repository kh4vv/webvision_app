import React, { Component} from "react";
import {Header, Image, Icon } from "semantic-ui-react";
//import SignatureCanvas from "react-signature-canvas";
import "bootstrap/dist/css/bootstrap.min.css";

import Upload from "../imageClassification/mnist_upload";
import Pad from "./pad";
import "./mnist.css";

class Mnist extends Component {

  state = {
    fn: null,
    predic: null,
  };

  render() {
    return (
      <div className="container" style={{ width: "800px" }}>
        <div style={{ margin: "20px" }}>
          <Subject />
          <Header as="h4">
            <Icon name="upload" />
            <Header.Content> Upload Mnist Image </Header.Content>
          </Header>
        </div>
        <Upload />
        <Image src={this.state.fn} style={imagestyle} />
        <br></br>
        <Header as="h2">
          <Icon name="plug" />
          <Header.Content> Prediction : {this.state.predic}</Header.Content>
        </Header>
        <Pad className="canvas" />
      </div>
    );
  }
}

const imagestyle = {
  height: "200px",
  width: "200px",
};

class Subject extends Component {
  render() {
    return (
      <header>
        <h2> MNIST Project</h2>
        <h3>
          {" "}
          You can either upload MNIST image or draw the numbers on the pad{" "}
        </h3>
      </header>
    );
  }
}

export default Mnist;
