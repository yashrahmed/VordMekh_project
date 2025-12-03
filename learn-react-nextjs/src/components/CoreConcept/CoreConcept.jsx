import "./CoreConcept.css";

export default function CoreConcept({image, title, description}) {
    return <li>
        <img src={image.src} alt={title}/>
        <h3>{title}</h3>
        <p>{description}</p>
    </li>
};