import React, {Component } from "react";
import Pad from "./pad";
import { Icon } from "semantic-ui-react";


class Quickdraw extends Component {
  render() {
    return (
      <div className="containter" style={{ width: "800px" }}>
        <Subject />
        <br></br>
        <Pad className="canvas" />
      </div>
    );
  }
}

class Subject extends Component {
  render() {
    return (
      <header>
        <h2>  Quickdraw Project</h2>
        <h3> <Icon name="pencil" />You can draw on the pad of anything below </h3>
        <h4>
          {" "}
          <Icon name="chess knight" />
          ant, bat, bear, bee, bird, butterfly, camel, cat, cow, dog, dolphin,
          dragon, duck, elephant, fish, flamingo, frog, giraffe, hedgehog,
          horse, kangaroo, lion, lobster, mermaid, monkey, mosquito, mouse,
          octopus, owl, panda, penguin, pig, rabbit, raccoon, shark, sheep,
          snail, snake, spider, squirrel, teddy-bear, tiger, whale, zebra
        </h4>
      </header>
    );
  }
}

export default Quickdraw;
