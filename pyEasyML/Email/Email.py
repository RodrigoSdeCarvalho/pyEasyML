

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from typing import Iterable, Optional, Tuple
import os
import ssl

def create_message(from_addr: str, to_addr: str, subject: str, 
                    message: str, attachments: Optional[Iterable[Tuple[str, str]]] = None,) -> MIMEMultipart:
    msg = MIMEMultipart()
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg["Subject"] = subject
    msg.attach(
        MIMEText(
            f"""\
    Hi, This is an automatic email sent from ADEG.

    {message}


    [do not reply]
        """,
                "plain",
        )
    )

    if attachments:
        for a_type, filepath in attachments:
            if a_type == "image":
                with open(filepath, "rb") as fp:
                    img_attchment = MIMEImage(fp.read(), name=os.path.basename(filepath))
                    img_attchment.add_header(
                        "Content-Disposition", "attachment", filename=os.path.basename(filepath)
                    )
                    msg.attach(img_attchment)
            else:
                with open(filepath, "r") as f:
                    txt_attchment = MIMEText(f.read())
                    txt_attchment.add_header(
                        "Content-Disposition", "attachment", filename=os.path.basename(filepath)
                    )
                    msg.attach(txt_attchment)

    return msg


def send_message(msg:MIMEMultipart, retries:int=5, from_password:str="") -> None:
    """Envia um email, a partir de uma conta gmail.

    Args:
        msg (MIMEMultipart): Mensagem, criada com a função create_message.
        retries (int, optional): Quantidade de reenvios, caso algum falhe (Não havendo falhas, enviar-se-ão todas as tentativas). Defaults to 5.
        from_password (str, optional): Senha da conta gmail que enviará o email. Defaults to "".
    """
    server = None
    for _ in range(0, retries):
        try:
            context = ssl.create_default_context()
            server = smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context)
            server.ehlo()
            server.login(msg["From"], from_password)
            server.send_message(msg, msg["From"], msg["To"])
            server.close()
        except Exception as e:
            print(e)
            if server:
                server.close()
