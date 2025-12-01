import { marked } from "marked";

marked.setOptions({
  breaks: true,
  gfm: true,
  highlight: function (code) {
    return code;
  }
});

export default marked;
