'use client';
import { useState } from "react";

import { CORE_CONCEPTS, EXAMPLES } from "@/resources/data";
import CoreConcept from "@/components/CoreConcept/CoreConcept";
import Header from "@/components/Header/Header";
import TabButton from "@/components/TabButton";

export function Home() {

    const [clickedButton, setClickedButton] = useState('');
    const topics = ["Components", "JSX", "Props", "State"];

    function clickHandler(selectedButton) {
        setClickedButton(selectedButton);
    }

    return (
        <div>
            <Header/>
            <main>
            <section id="core-concepts">
                <ul>
                    {CORE_CONCEPTS.map(item => <CoreConcept image={item.image} key={item.title} title={item.title} description={item.description}/>)}
                </ul>
            </section>
            <section id="examples">
                <h2>Examples</h2>
                <menu>
                    {
                        topics.map(topic => 
                            <TabButton
                                key={topic} 
                                onClick={() => clickHandler(topic.toLowerCase())} 
                                isSelected={clickedButton==topic.toLowerCase()}>
                                    {topic}
                            </TabButton>)
                    }
                </menu>
                {clickedButton == '' &&  <div id="tab-content">Please select a topic</div>}
                {clickedButton != '' &&  
                    <div id="tab-content">
                        <h3>{EXAMPLES[clickedButton].title}</h3>
                        <p>{EXAMPLES[clickedButton].description}</p>
                        <pre>
                            <code>
                                {EXAMPLES[clickedButton].code}
                            </code>
                        </pre>
                    </div>
                }
            </section>
            </main>
        </div>
    );
}
