PGDMP                  
    |            license-plate    16.4    16.4     �           0    0    ENCODING    ENCODING        SET client_encoding = 'UTF8';
                      false            �           0    0 
   STDSTRINGS 
   STDSTRINGS     (   SET standard_conforming_strings = 'on';
                      false            �           0    0 
   SEARCHPATH 
   SEARCHPATH     8   SELECT pg_catalog.set_config('search_path', '', false);
                      false            �           1262    16398    license-plate    DATABASE     �   CREATE DATABASE "license-plate" WITH TEMPLATE = template0 ENCODING = 'UTF8' LOCALE_PROVIDER = libc LOCALE = 'Thai_Thailand.1252';
    DROP DATABASE "license-plate";
                postgres    false            �            1259    16400    plateDetection    TABLE     �   CREATE TABLE public."plateDetection" (
    id integer NOT NULL,
    f_image text,
    p_image text,
    p_text text,
    province text,
    date timestamp with time zone
);
 $   DROP TABLE public."plateDetection";
       public         heap    postgres    false            �            1259    16399    plateDetection_id_seq    SEQUENCE     �   CREATE SEQUENCE public."plateDetection_id_seq"
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
 .   DROP SEQUENCE public."plateDetection_id_seq";
       public          postgres    false    216            �           0    0    plateDetection_id_seq    SEQUENCE OWNED BY     S   ALTER SEQUENCE public."plateDetection_id_seq" OWNED BY public."plateDetection".id;
          public          postgres    false    215            P           2604    16403    plateDetection id    DEFAULT     z   ALTER TABLE ONLY public."plateDetection" ALTER COLUMN id SET DEFAULT nextval('public."plateDetection_id_seq"'::regclass);
 B   ALTER TABLE public."plateDetection" ALTER COLUMN id DROP DEFAULT;
       public          postgres    false    216    215    216            �          0    16400    plateDetection 
   TABLE DATA           X   COPY public."plateDetection" (id, f_image, p_image, p_text, province, date) FROM stdin;
    public          postgres    false    216   |       �           0    0    plateDetection_id_seq    SEQUENCE SET     G   SELECT pg_catalog.setval('public."plateDetection_id_seq"', 163, true);
          public          postgres    false    215            R           2606    16408 "   plateDetection plateDetection_pkey 
   CONSTRAINT     d   ALTER TABLE ONLY public."plateDetection"
    ADD CONSTRAINT "plateDetection_pkey" PRIMARY KEY (id);
 P   ALTER TABLE ONLY public."plateDetection" DROP CONSTRAINT "plateDetection_pkey";
       public            postgres    false    216            �   C  x��ӿJA�:��%�����gB�gD{�G�4���m�Q<�j�9�����)~��x�������궮��3��f�w[�/���r��7����;A**����{�wQy��;���ʫʇʗ�^���E��4sv
�!dDC�tfC�w���T&�h��j��������N��U���-p����-nL.U�TQ��L��������@��b'���|�,��(��I)AQ�N���U��� 2��ꉸ�t�;�g��C*	2���'US+o�O�|Q�x+���m4��XTn�<��j%�"w|�SU��тb     