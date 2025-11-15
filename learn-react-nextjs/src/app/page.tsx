import Image from "next/image";
import styles from "./page.module.css";

function VercelImg() {
  return <Image
    className={styles.logo}
    src="/next.svg"
    alt="Next.js logo"
    width={100}
    height={20}
    priority
  />;
}

export default function Home() {
  return (
    <div className={styles.page}>
      <main className={styles.main}>
        <VercelImg />
        <div className={styles.intro}>
          <h1>To get started, edit the page.tsx file.</h1>
        </div>
      </main>
    </div>
  );
}
