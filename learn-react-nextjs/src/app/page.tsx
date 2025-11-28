import LikeButton from './like-button';

function ListOfNames() {
    const names = ['Ada Lovelace', 'Grace Hopper', 'Margaret Hamilton'];

    return <div>
        <ul>
            {
                names.map((name) => (
                <li key={name}>
                    {name} 
                </li>))
            }
        </ul>
        <LikeButton/>
    </div>;
}

function Header(props) {// all props imported as a dictionary.
    let {title} = props // ES destructuring.
    console.log(title); 
    return <h1>Developer Preview for {title}!</h1>;
}

export default function HomePage() {
    return <div>
        <Header title="React Tutorial"/>
        <p>Here is some content</p>
        <ListOfNames/>
    </div>;
}
