import React, { Component} from "react";
import {Header, Image, Icon } from "semantic-ui-react";
import "bootstrap/dist/css/bootstrap.min.css";

import Upload from "./landmark_upload";
import "./mnist.css";

class LandMark extends Component {

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
            <Header.Content> Upload Korea Landmark Image </Header.Content>
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
        <h2> Korea Landmark Project</h2>
        <h3>
          {" "}
          You can upload Korea Landmark image{" "}
        </h3>
      </header>
    );
  }
}

export default LandMark;