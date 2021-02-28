import React, { Component} from "react";
import {Header, Image, Icon } from "semantic-ui-react";
import "bootstrap/dist/css/bootstrap.min.css";

import Upload from "./yolov3_upload";

class Yolov3 extends Component {

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
            <Header.Content> Upload Any Image </Header.Content>
          </Header>
        </div>
        <Upload />
        <Image src={this.state.fn} style={imagestyle} />
        <br></br>
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
        <h2> YOLO v3  Project</h2>
        <h3>
          {" "}
          You can upload image - return boundry boxes with labels{" "}
        </h3>
      </header>
    );
  }
}

export default Yolov3;