import List from "./List";

const synonyms = ["essentials", "fundamentals", "core concepts"]

function getRandomSyn() {
  let idx = Math.floor(Math.random() * synonyms.length);
  return synonyms[idx];
}

export default function Home() {
  return (
    <div>
       This is the homepage for React {getRandomSyn()}.
       <List/>
    </div>
  );
}
