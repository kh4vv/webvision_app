import "./App.css";
import React, { Component } from "react";
import { BrowserRouter as Router, Route, Switch } from "react-router-dom";
import { Grid } from "semantic-ui-react";
import "bootstrap/dist/css/bootstrap.min.css";
//Image Classification
import mnist from "./imageClassification/mnist";
import quickdraw from "./imageClassification/quickdraw";
import landmark from "./imageClassification/landmark";
//Object detection
import Yolov3 from "./objectDetection/yolov3"
import main from "./components/welcome";
import Subject from "./components/subject";
import Footer from "./components/footer";
import AccordionStyled from "./components/accodion";

class App extends Component {
  render() {
    return (
      <div className="App">
        <Subject></Subject>
        <Grid columns={2}>
          <Grid.Column width={3}>
            <AccordionStyled></AccordionStyled>
            <Footer className="footer" />
          </Grid.Column>
          <Grid.Column width={11}>
            <Router>
              <article>
                <Switch>
                  <Route path="/welcome" component={main} />
                  <Route path="/mnist" component={mnist} />
                  <Route path="/quickdraw" component={quickdraw} />
                  <Route path="/landmark" component={landmark} />
                  <Route path="/yolov3" component={Yolov3} />
                </Switch>
              </article>
            </Router>
          </Grid.Column>
        </Grid>
      </div>
    );
  }
}

export default App;
