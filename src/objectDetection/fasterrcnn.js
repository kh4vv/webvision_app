import React, { Component } from "react";
import { Header, Icon } from "semantic-ui-react";
import "bootstrap/dist/css/bootstrap.min.css";

import Upload from "./fasterrcnn_upload";

class Yolov3 extends Component {

  render() {
    return (
      <div className="container" style={{ width: "800px" }}>
        <div style={{ margin: "20px" }}>
          <Subject />
          <Header as="h3">
            <Icon name="upload" />
            <Header.Content> Upload Any Image </Header.Content>
          </Header>
        </div>
        <Upload />
        <br></br>
      </div>
    );
  }
}

class Subject extends Component {
  render() {
    return (
      <header>
        <h2> <Icon name="camera retro" /> Faster RCNN Project</h2>
        <h3>
          {" "}
          You can upload image - return boundry boxes with labels{" "}
        </h3>
      </header>
    );
  }
}

export default Yolov3;