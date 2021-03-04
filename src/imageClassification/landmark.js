import React, { Component } from "react";
import { Header, Icon } from "semantic-ui-react";

import Upload from "./landmark_upload";

class LandMark extends Component {

  render() {
    return (
      <div className="container" style={{ width: "800px" }}>
        <div style={{ margin: "20px" }}>
          <Subject />
          <Header as="h4">
            <Icon name="upload" />
            <Header.Content> Upload Korea Landmark Image </Header.Content>
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
        <h2> <Icon name="globe" />Korea Landmark Project</h2>
        <h3>
          {" "}
          You can upload Korea Landmark image{" "}
        </h3>
      </header>
    );
  }
}

export default LandMark;