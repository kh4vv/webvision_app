import React, { Component } from "react";
import { Header, Icon } from "semantic-ui-react";

import Upload from "./maskrcnn_upload";

class Maskrcnn extends Component {

  render() {
    return (
      <div className="container" style={{ width: "800px" }}>
        <div style={{ margin: "20px" }}>
          <Subject />
          <Header as="h4">
            <Icon name="upload" />
            <Header.Content> Upload Image </Header.Content>
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
        <h2> <Icon name="unhide" /> MASK RCNN Project</h2>
        <h3>
          {" "}
          You can upload any photo and return bounding boxes with mask {" "}
        </h3>
      </header>
    );
  }
}

export default Maskrcnn;
