'use client';

import { useState } from "react";

export default function LikeButton() {
    const [likes, setLikes] = useState(0);

    function likeClickHandler() {
        setLikes(likes + 1);
    }

    return <>
        <button onClick={likeClickHandler}>Like this article!</button>
        <p>Liked by {likes} people!</p>
    </>;
}