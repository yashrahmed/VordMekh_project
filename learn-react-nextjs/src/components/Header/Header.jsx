import reactImg from "@/resources/assets/react-core-concepts.png";
import "./Header.css";

const synonyms = ["essentials", "fundamentals", "core concepts"]

function getRandomSyn() {
  // let idx = Math.floor(Math.random() * synonyms.length);
  return synonyms[1]; // Hardcoding to 1 as a random index leads to hydration diff errors when using next js.
}

export default function Header() {
  return (
    <header>
      <h1>React Essentials</h1>
        <img src={reactImg.src} alt="Stylized atom"/>
       <p>
        This is the homepage for React {getRandomSyn()}.
      </p>
    </header>);
}