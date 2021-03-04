import React, { Component } from "react";
import { Header, Icon } from "semantic-ui-react";

import Upload from "../imageClassification/mnist_upload";
import Pad from "./mnist_pad";

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
        <br></br>
        <Pad style={{ height: "300px", width: "300px", border: "2px solid #F0F" }} />
      </div>
    );
  }
}

class Subject extends Component {
  render() {
    return (
      <header>
        <h2> <Icon name="numbered list" /> MNIST Project</h2>
        <h3>
          {" "}
          You can either upload MNIST image or draw the numbers on the pad{" "}
        </h3>
      </header>
    );
  }
}

export default Mnist;
